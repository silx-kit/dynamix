      subroutine fecorrt(pix,t,cor,lenpi,lpixels)
      integer lenpi, lpixels, i, t0, j
      integer pix(0:lenpi-1), t(0:lenpi-1), cor(0:lpixels-1,0:lpixels-1)
Cf2py intent(in) pix
Cf2py intent(in) t
Cf2py intent(in,out) cor
C     fecorrt function to caclulate 2t cf from events
      i=0
      do while (i .LT. lenpi) 
          t0=t(i)
          j=i+1
          do while (pix(j) .EQ. pix(i)) 
              cor(t(j),t0)=cor(t(j),t0)+1
              j=j+1
              if (j .GE. lenpi) exit
          enddo
          i=i+1
      enddo
      do 20 i=0,lpixels-1 
        do 10 j=i,lpixels-1
          cor(j,i)=(cor(j,i)+cor(i,j))
          cor(i,j)=cor(j,i)  
10      continue
20    continue          
      return
      end
