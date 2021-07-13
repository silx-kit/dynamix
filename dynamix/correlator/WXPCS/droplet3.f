      Subroutine droplet3(pix,ip,matr,dar,thr,nx,ny,lp)
      integer nx, ny, lp,ix, iy, nxx, nyy,st,ip,jj
      integer matro(0:nx-1,0:ny-1),cxj,cyj,iw 
      integer j,cx(0:lp-1),cy(0:lp-1),jjj
      real fs,thr
      real matr(0:nx-1,0:ny-1),dar(0:nx-1,0:ny-1)
      real pix(0:lp-1)
Cf2py intent(out) pix
Cf2py intent(out) ip
Cf2py intent(in,c) matr
Cf2py intent(in,c) dar
Cf2py intent(in) thr
Cf2py intent(in) lp
Cf2py depend(nx,ny) matr
Cf2py depend(nx,ny) dar
Cf2py depend(lp) pix

C     this is to find droplets really good program!!!      
      j=0
      ip=0
      nxx=nx-1 
      nyy=ny-1
      fs=0 
      st=-1 
      iw=-1 
      do 42 iy=0,nyy,1
       do 41 ix=0,nxx,1
        matr(ix,iy)=matr(ix,iy)-dar(ix,iy)
        if (matr(ix,iy) .LT. thr) then 
         matr(ix,iy)=0.0
        endif
        matro(ix,iy)=0
41     continue
42    continue       
      do 40 iy=0,nyy,1
       do 30 ix=0,nxx,1
        if (matr(ix,iy) .GT. 0.0 .AND. matro(ix,iy) .EQ. 0) then
         matro(ix,iy)=1
         st=0
         cx(j)=ix
         cy(j)=iy
         iw=-1
        endif  
        do while (st .GE. 0)
         iw=iw+1
         cxj=cx(iw)
         cyj=cy(iw)  
         st=st-1
         if (cxj-1 .GE. 0) then
          if (matr(cxj-1,cyj) .GT. 0.0 .AND. matro(cxj-1,cyj)
     1       .EQ. 0) then      
             matro(cxj-1,cyj)=1
             j=j+1
            cx(j)=cxj-1
            cy(j)=cyj
            st=st+1 
          endif
         endif
         if (cxj+1 .LE. nxx) then
          if (matr(cxj+1,cyj) .GT. 0.0 .AND. matro(cxj+1,cyj) 
     1       .EQ. 0) then      
             matro(cxj+1,cyj)=1
             j=j+1
             cx(j)=cxj+1
             cy(j)=cyj 
             st=st+1
          endif
         endif
         if (cyj-1 .GE. 0) then 
          if (matr(cxj,cyj-1) .GT. 0.0 .AND. matro(cxj,cyj-1) 
     1       .EQ. 0) then      
             matro(cxj,cyj-1)=1
             j=j+1
             cx(j)=cxj
             cy(j)=cyj-1  
             st=st+1  
          endif
         endif
         if (cyj+1 .LE. nyy) then
          if (matr(cxj,cyj+1) .GT. 0.0 .AND. matro(cxj,cyj+1)
     1       .EQ. 0) then      
             matro(cxj,cyj+1)=1
             j=j+1
             cx(j)=cxj
             cy(j)=cyj+1 
             st=st+1 
          endif
         endif
        enddo
        if (iw .GE. 0) then
         do 20 jjj=0,iw,1
          fs=fs+matr(cx(jjj),cy(jjj))
          cx(jjj)=0
          cy(jjj)=0  
20       continue
         pix(ip)=fs
         ip=ip+1 
        endif
       fs=0
       j=0
       jj=0
       st=-1
       iw=-1 
30     continue
40    continue
      return 
      end Subroutine
