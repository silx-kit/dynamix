      Subroutine dropimgood(coun,pix,dar,mt,dk,thr,lt,ut,lp,ph,nx,ny)
      integer pix(lp),coun,tmp,dar(nx,ny),ifs
      integer nx, ny,ix, iy,st,mx,my,ph,ixx,iyy,bx,ex,by,ey
      integer mto(nx,ny),cxj,cyj,iw,lp,ut 
      integer j,cx(lp),cy(lp),jjj
      real fss,thr,mpx,dk(nx,ny)
      real mt(nx,ny),rfs
Cf2py intent(out) coun
Cf2py intent(out) pix
Cf2py intent(out,c) dar
Cf2py intent(in,c) mt
Cf2py intent(in,c) dk
Cf2py intent(in) thr
Cf2py intent(in) lt
Cf2py intent(in) ut
Cf2py intent(in) lp
Cf2py intent(in) ph
Cf2py depend(nx,ny) dar
Cf2py depend(nx,ny) mt
Cf2py depend(nx,ny) dk

C     this is to dropletize the hole image!!!      
C     for wxpcs
C     msumpix,mpix,tmp=dropimgood(mt,dkimg,lth,bADU,tADU,mNp,ph,nx,ny)
      coun=0
      j=1
      fss=0.0
      mpx=0.0 
      mx=0
      my=0 
      st=-1 
      iw=0 
      do 42 iy=1,ny,1
       do 41 ix=1,nx,1
        mt(ix,iy)=mt(ix,iy)-dk(ix,iy)
        dar(ix,iy)=0 
        mto(ix,iy)=0
        if (mt(ix,iy) .LT. thr) then 
          mt(ix,iy)=0.0
         mto(ix,iy)=1
        endif
C        if (mt(ix,iy) .GE. lt .AND. mt(ix,iy) .LE. ut) then
C         dar(ix,iy)=1
C         mt(ix,iy)=mt(ix,iy)-ph
C         mto(ix,iy)=1
C        endif 
C        if (mt(ix,iy) .GT. ut) then
C         dar(ix,iy)=nint(mt(ix,iy)/ph)
C         mt(ix,iy)=mt(ix,iy)-ph*dar(ix,iy)
C         mto(ix,iy)=1
C        endif 
41     continue
42    continue       
      do 40 iy=1,ny,1
       do 30 ix=1,nx,1
        if (mto(ix,iy) .EQ. 0) then
         mto(ix,iy)=1
         st=0
         cx(j)=ix
         cy(j)=iy
         iw=0
        endif  
        do while (st .GE. 0)
         iw=iw+1
         cxj=cx(iw)
         cyj=cy(iw)  
         st=st-1
         if (cxj-1 .GE. 1) then
          if (mto(cxj-1,cyj) .EQ. 0) then      
            mto(cxj-1,cyj)=1
            j=j+1
            cx(j)=cxj-1
            cy(j)=cyj
            st=st+1 
          endif
         endif
         if (cxj+1 .LE. nx) then
          if (mto(cxj+1,cyj) .EQ. 0) then      
             mto(cxj+1,cyj)=1
             j=j+1
             cx(j)=cxj+1
             cy(j)=cyj 
             st=st+1
          endif
         endif
         if (cyj-1 .GE. 1) then 
          if (mto(cxj,cyj-1) .EQ. 0) then      
             mto(cxj,cyj-1)=1
             j=j+1
             cx(j)=cxj
             cy(j)=cyj-1  
             st=st+1  
          endif
         endif
         if (cyj+1 .LE. ny) then
          if (mto(cxj,cyj+1) .EQ. 0) then      
             mto(cxj,cyj+1)=1
             j=j+1
             cx(j)=cxj
             cy(j)=cyj+1 
             st=st+1 
          endif
         endif
        enddo
        if (iw .GE. 1) then
         rfs=0
         do 19 jjj=1,iw,1
          rfs=rfs+mt(cx(jjj),cy(jjj))
19       continue
         ifs=nint(rfs/ph)
         do while (ifs .GT. 0)
C          print*,'iw',iw
          mpx=0.0
          do 20 jjj=1,iw,1
           if (mt(cx(jjj),cy(jjj)) .GT. mpx) then
            mpx=mt(cx(jjj),cy(jjj))
            mx=cx(jjj)
            my=cy(jjj)
           endif
20        continue
          if (mx .EQ. 1) then
            bx=mx
            ex=mx+1      
          elseif (mx .EQ. nx) then
            bx=mx-1
            ex=mx
          else
            bx=mx-1
            ex=mx+1
          endif
          if (my .EQ. 1) then
            by=my
            ey=my+1
          elseif (my .EQ. ny) then
            by=my-1
            ey=my
          else
            by=my-1
            ey=my+1
          endif
          fss=0.0
          do 7 iyy=by,ey,1
           do 6 ixx=bx,ex,1
             fss=fss+mt(ixx,iyy)
6          continue
7         continue
C          print*,'fss',fss
          if (fss .GE. lt .AND. fss .LE. ut) then 
            dar(mx,my)=1
            mt(mx,my)=0.0
C            mto(mx,my)=1
            ifs=ifs-1
            rfs=rfs-ph
          elseif (fss .GT. ut) then
            if (mpx .LE. ut) then 
             dar(mx,my)=1
             mt(mx,my)=0.0
C             mto(mx,my)=1
            elseif (mpx .GT. ut) then 
             dar(mx,my)=nint(mpx/ph)
             mt(mx,my)=mt(mx,my)-dar(mx,my)*ph
C             mto(mx,my)=1
            endif 
            ifs=ifs-dar(mx,my)
            rfs=rfs-ph*dar(mx,my) 
          else
C            print*,rfs,fss,ifs,mpx
            if (rfs .LT. lt .OR. mpx .LE. 0.0) then 
C            if (mpx .LE. 0.0) then
             mt(mx,my)=0.0
C             mto(mx,my)=1
             ifs=0
            else
             mt(mx,my)=0.0
C             mto(mx,my)=1
            endif
          endif 
         enddo
         do 21 jjj=1,iw,1
          mt(cx(jjj),cy(jjj))=0.0
          mto(cx(jjj),cy(jjj))=1
          cx(jjj)=0
          cy(jjj)=0  
21       continue  
        endif
       j=1
       st=-1
       iw=0 
30     continue
40    continue
      do 60 iy=1,ny,1
       do 50 ix=1,nx,1
        tmp=dar(ix,iy)
        do while (tmp .GT. 0)
          coun=coun+1   
          pix(coun)=nx*iy+ix
          tmp=tmp-1 
        enddo
50     continue
60    continue
      return 
      end Subroutine
