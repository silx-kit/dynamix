      subroutine fecorr(pix,t,cor,lenpi,lpixels)
      integer lenpi, lpixels, i, t0, j, ti
      integer pix(0:lenpi-1), t(0:lenpi-1), cor(0:lpixels-1)
Cf2py intent(in) pix
Cf2py intent(in) t
Cf2py intent(in,out) cor
C     fecorr function to calculate cf from events
      i=0
      do while (i .LT. lenpi) 
          t0=t(i)
          j=i+1
          do while (pix(j) .EQ. pix(i)) 
              ti=abs(t(j)-t0)
              cor(ti)=cor(ti)+1
              j=j+1
              if (j .GE. lenpi) exit
          enddo
          i=i+1
      enddo
      return
      end
