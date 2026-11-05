// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// Cooked polygon clipping in the existing sparse Sc/PCM continuation.
// Iterate explicit boundary edges without bounded private polygon arrays.
struct ConvexClip {
    GjkBoxPair pair;
    uint referenceB,referenceFace,incidentFace,nr,ni;
    float3 referenceNormal,incidentNormal,anchor,incidentAnchor;
    float contactDistance,tolerance,scale;
};

float3 convexAxis(uint i) {
    return i==0u?float3(1.f,0.f,0.f):
           i==1u?float3(0.f,1.f,0.f):float3(0.f,0.f,1.f);
}
float convexComponent(float3 v,uint i) {
    return i==0u?v.x:(i==1u?v.y:v.z);
}
uint convexFaceCount(GjkBoxPair pair,bool first) {
    const uint kind=first?pair.kindA:pair.kindB;
    if(kind!=2u)return 6u;
    const uint vertex=first?pair.firstA:pair.firstB;
    return hullSurfaces.Load(
        p.surfaceBias+vertex*8u+4u);
}
uint convexFaceBase(GjkBoxPair pair,bool first) {
    const uint vertex=first?pair.firstA:pair.firstB;
    return hullSurfaces.Load(p.surfaceBias+vertex*8u);
}
uint convexFaceAddress(GjkBoxPair pair,bool first,uint face) {
    return p.surfaceBias+p.surfaceFacesBase+
        (convexFaceBase(pair,first)+face)*24u;
}
float3 convexFaceNormalLocal(GjkBoxPair pair,bool first,uint face) {
    const uint kind=first?pair.kindA:pair.kindB;
    if(kind==2u)
        return asfloat(hullSurfaces.Load3(
            convexFaceAddress(pair,first,face)));
    return convexAxis(face/2u)*((face&1u)!=0u?1.f:-1.f);
}
float3 convexFaceNormalB(GjkBoxPair pair,bool first,uint face) {
    const float3 normal=convexFaceNormalLocal(pair,first,face);
    return first?rotateSource(pair.relative,normal):normal;
}
uint convexFaceVertexCount(GjkBoxPair pair,bool first,uint face) {
    const uint kind=first?pair.kindA:pair.kindB;
    return kind==2u
        ?hullSurfaces.Load(convexFaceAddress(pair,first,face)+20u):4u;
}
float3 convexFaceVertex(GjkBoxPair pair,bool first,
                        uint face,uint ordinal) {
    const uint kind=first?pair.kindA:pair.kindB;
    float3 vertex;
    if(kind==2u) {
        const uint address=convexFaceAddress(pair,first,face);
        const uint start=hullSurfaces.Load(address+16u);
        const uint vertexIndex=hullSurfaces.Load(
            p.surfaceBias+p.surfaceIndicesBase+
            (start+ordinal)*4u);
        const uint firstVertex=first?pair.firstA:pair.firstB;
        vertex=asfloat(hullVertices.Load3(
            p.hullBias+(firstVertex+vertexIndex)*12u));
    }else {
        const uint coordinate=face/2u;
        const float3 n=convexFaceNormalLocal(pair,first,face);
        const float3 u=convexAxis((coordinate+1u)%3u);
        const float3 v=cross(n,u);
        const float3 half=first?pair.halfA:pair.halfB;
        const float su=(ordinal==0u || ordinal==3u)?1.f:-1.f;
        const float sv=ordinal<2u?1.f:-1.f;
        vertex=n*convexComponent(half,coordinate)+
            u*(su*convexComponent(half,(coordinate+1u)%3u))+
            v*(sv*convexComponent(half,(coordinate+2u)%3u));
    }
    return first?pair.aInB+rotateSource(pair.relative,vertex):vertex;
}
float3 convexReferenceVertex(ConvexClip c,uint i) {
    const bool first=c.referenceB==0u;
    return convexFaceVertex(c.pair,first,c.referenceFace,i%c.nr);
}
float3 convexIncidentVertex(ConvexClip c,uint i) {
    const bool first=c.referenceB!=0u;
    return convexFaceVertex(c.pair,first,c.incidentFace,i%c.ni);
}
float convexCross2(ConvexClip c,float3 a,float3 b) {
    return gjkDot(c.referenceNormal,cross(a,b));
}
bool convexInsideReference(ConvexClip c,float3 probe) {
    for(uint i=0u;i<c.nr;++i) {
        const float3 a=convexReferenceVertex(c,i);
        const float3 b=convexReferenceVertex(c,i+1u);
        if(convexCross2(c,b-a,probe-a)<-c.tolerance*c.scale)
            return false;
    }
    return true;
}
bool convexInsideIncident(ConvexClip c,float3 probe) {
    for(uint i=0u;i<c.ni;++i) {
        const float3 a=convexIncidentVertex(c,i);
        const float3 b=convexIncidentVertex(c,i+1u);
        if(convexCross2(c,b-a,probe-a)>c.tolerance*c.scale)
            return false;
    }
    return true;
}
void convexEmit(ConvexClip c,float3 incidentPoint,
                inout GjkWitness contacts[4],inout uint count) {
    const float separation=gjkDot(c.referenceNormal,
        incidentPoint-c.anchor);
    if(!isfinite(separation) || separation>c.contactDistance)return;
    const float3 projected=incidentPoint-c.referenceNormal*separation;
    const float3 a=c.referenceB!=0u?incidentPoint:projected;
    const float3 b=c.referenceB!=0u?projected:incidentPoint;
    GjkWitness candidate=(GjkWitness)0;
    candidate.localA=rotateSource(
        conjugate(c.pair.relative),a-c.pair.aInB);
    candidate.localB=b;
    candidate.normalB=c.referenceB!=0u
        ?c.referenceNormal:-c.referenceNormal;
    candidate.separation=separation;
    if(!all(isfinite(candidate.localA)) ||
       !all(isfinite(candidate.localB)))return;
    for(uint i=0u;i<count;++i) {
        if(gjkLength2(contacts[i].localB-candidate.localB)<=
           c.tolerance*c.tolerance) {
            if(candidate.separation<contacts[i].separation)
                contacts[i]=candidate;
            return;
        }
    }
    if(count<4u) {
        contacts[count++]=candidate;
        return;
    }
    GjkWitness choices[5]={
        contacts[0],contacts[1],contacts[2],contacts[3],candidate};
    uint deepest=0u;
    for(uint i=1u;i<5u;++i)
        if(choices[i].separation<choices[deepest].separation)
            deepest=i;
    float best=-1.f;
    uint omit=4u;
    for(uint excluded=0u;excluded<5u;++excluded) {
        if(excluded==deepest)continue;
        float score=0.f;
        for(uint i=0u;i<5u;++i) {
            if(i==excluded)continue;
            for(uint j=i+1u;j<5u;++j)
                if(j!=excluded)
                    score+=gjkLength2(
                        choices[i].localB-choices[j].localB);
        }
        if(score>best){best=score;omit=excluded;}
    }
    if(omit!=4u)contacts[omit]=candidate;
}
uint convexFaceContacts(GjkBoxPair pair,float contactDistance,
                        GjkWitness witness,
                        out GjkWitness contacts[4]) {
    if(p.surfaceBytes==0u ||
       (pair.kindA!=2u && pair.kindB!=2u) ||
       pair.kindA==1u || pair.kindB==1u)
        return 0u;
    const float3 normal=witness.normalB;
    if(!(gjkLength2(normal)>0.f))return 0u;
    uint faceA=0u,faceB=0u;
    float alignmentA=-2.f,alignmentB=-2.f;
    const uint facesA=convexFaceCount(pair,true);
    const uint facesB=convexFaceCount(pair,false);
    for(uint i=0u;i<facesA;++i) {
        const float d=gjkDot(-normal,convexFaceNormalB(pair,true,i));
        if(d>alignmentA){alignmentA=d;faceA=i;}
    }
    for(uint i=0u;i<facesB;++i) {
        const float d=gjkDot(normal,convexFaceNormalB(pair,false,i));
        if(d>alignmentB){alignmentB=d;faceB=i;}
    }
    if(max(alignmentA,alignmentB)<0.95f)return 0u;
    ConvexClip c=(ConvexClip)0;
    c.pair=pair;
    c.referenceB=alignmentB>=alignmentA?1u:0u;
    c.referenceFace=c.referenceB!=0u?faceB:faceA;
    const bool referenceFirst=c.referenceB==0u;
    c.referenceNormal=convexFaceNormalB(
        pair,referenceFirst,c.referenceFace);
    const uint incidentCount=convexFaceCount(pair,!referenceFirst);
    float incidentAlignment=-2.f;
    for(uint i=0u;i<incidentCount;++i) {
        const float d=gjkDot(-c.referenceNormal,
            convexFaceNormalB(pair,!referenceFirst,i));
        if(d>incidentAlignment) {
            incidentAlignment=d;
            c.incidentFace=i;
        }
    }
    c.incidentNormal=convexFaceNormalB(
        pair,!referenceFirst,c.incidentFace);
    const float denominator=gjkDot(
        c.incidentNormal,c.referenceNormal);
    if(!(denominator< -0.25f))return 0u;
    c.nr=convexFaceVertexCount(
        pair,referenceFirst,c.referenceFace);
    c.ni=convexFaceVertexCount(
        pair,!referenceFirst,c.incidentFace);
    if(c.nr<3u || c.ni<3u)return 0u;
    c.anchor=convexReferenceVertex(c,0u);
    c.incidentAnchor=convexIncidentVertex(c,0u);
    float edgeLengthSq=0.f;
    for(uint i=0u;i<c.nr;++i) {
        const float3 a=convexReferenceVertex(c,i);
        const float3 b=convexReferenceVertex(c,i+1u);
        edgeLengthSq=max(edgeLengthSq,gjkLength2(b-a));
    }
    for(uint i=0u;i<c.ni;++i) {
        const float3 a=convexIncidentVertex(c,i);
        const float3 b=convexIncidentVertex(c,i+1u);
        edgeLengthSq=max(edgeLengthSq,gjkLength2(b-a));
    }
    c.scale=sqrt(edgeLengthSq);
    c.tolerance=max(c.scale*1.e-5f,1.e-6f);
    c.contactDistance=contactDistance;
    uint count=0u;
    for(uint i=0u;i<c.ni;++i) {
        const float3 v=convexIncidentVertex(c,i);
        if(convexInsideReference(c,v))
            convexEmit(c,v,contacts,count);
    }
    for(uint i=0u;i<c.nr;++i) {
        const float3 v=convexReferenceVertex(c,i);
        if(!convexInsideIncident(c,v))continue;
        const float height=gjkDot(c.incidentNormal,
            c.incidentAnchor-v)/denominator;
        if(isfinite(height))
            convexEmit(c,v+c.referenceNormal*height,contacts,count);
    }
    for(uint i=0u;i<c.ni;++i) {
        const float3 a=convexIncidentVertex(c,i);
        const float3 e=convexIncidentVertex(c,i+1u)-a;
        for(uint j=0u;j<c.nr;++j) {
            const float3 b=convexReferenceVertex(c,j);
            const float3 f=convexReferenceVertex(c,j+1u)-b;
            const float determinant=convexCross2(c,e,f);
            if(abs(determinant)<=c.tolerance*c.tolerance)continue;
            const float3 d=b-a;
            const float t=convexCross2(c,d,f)/determinant;
            const float u=convexCross2(c,d,e)/determinant;
            if(t>=-c.tolerance && t<=1.f+c.tolerance &&
               u>=-c.tolerance && u<=1.f+c.tolerance)
                convexEmit(c,a+e*clamp(t,0.f,1.f),contacts,count);
        }
    }
    return count;
}
