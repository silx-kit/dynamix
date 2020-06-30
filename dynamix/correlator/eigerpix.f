      Subroutine eigerpix(coun,pix,mt,lp,nx)
      integer pix(lp),coun,tmp
      integer nx, ix,lp
      integer mt(nx)
Cf2py intent(out) coun
Cf2py intent(out) pix
Cf2py intent(in,c) mt
Cf2py intent(in) lp
Cf2py depend(nx) dar
Cf2py depend(nx) mt

C     this is to dropletize the hole image!!!      
C     for wxpcs and Eiger or Maxipix detector
C     msumpix,mpix=eigerpix(mt,mNp,nx)
      coun=0
      do 50 ix=1,nx,1
       tmp=mt(ix)
       do while (tmp .GT. 0)
         coun=coun+1
         pix(coun)=ix
         tmp=tmp-1 
       enddo
50    continue
      return 
      end Subroutine
