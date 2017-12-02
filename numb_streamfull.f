c   compilation:  f2py -m numb_streamfull -c --fcompiler=gnu95 numb_streamfull.f
c--------------------------------------------------------------------------
c----------------------------------------------------------------------------
      subroutine det3(v1,v2,v3,d3)
c   d3 = det3(v1,v2,v3)
	  IMPLICIT NONE
Cf2py intent(in):: v1
Cf2py intent(in):: v2
Cf2py intent(in):: v3
Cf2py intent(out):: d3
      real(8),dimension(0:2):: v1,v2,v3
      real(8) :: d3
      !write(*,*) ' det3------------------'
      !write(*,*) v1
      !write(*,*) v2
      !write(*,*) v3
      d3 = v1(0)*v2(1)*v3(2)+v1(1)*v2(2)*v3(0)+v1(2)*v2(0)*v3(1)
      d3 = d3 - v1(2)*v2(1)*v3(0)-v1(0)*v2(2)*v3(1)-v1(1)*v2(0)*v3(2)
      end subroutine det3
c-------------------------------------------------------------------------     
      subroutine det4(v1,v2,v3,v4,d4)
c   d4 = det4(v1,v2,v3,v4)
	  IMPLICIT NONE
Cf2py intent(in):: v1
Cf2py intent(in):: v2
Cf2py intent(in):: v3
Cf2py intent(in):: v4
Cf2py intent(out):: d4
      real(8), dimension(0:2):: v1,v2,v3,v4
      real(8) :: d4,d31,d32,d33,d34
      !write(*,*) ' det4----------------------------'
      !write(*,*) v1
      !write(*,*) v2
      !write(*,*) v3
      !write(*,*) v4
      call det3(v2,v3,v4,d31) 
      call det3(v1,v3,v4,d32)
      call det3(v1,v2,v4,d33)
      call det3(v1,v2,v3,d34)
      d4 = -d31+d32-d33+d34
      end subroutine det4
c---------------------------------------------------------------- 
      subroutine point_in_tetr(v1,v2,v3,v4,vp,d5)
c   d5 = point_in_tetr(v1,v2,v3,v4,vp)
	  IMPLICIT NONE
      real(8), dimension(0:2):: v1,v2,v3,v4,vp
      real(8), dimension(0:4):: d5
Cf2py intent(in):: v1
Cf2py intent(in):: v2
Cf2py intent(in):: v3
Cf2py intent(in):: v4
Cf2py intent(in):: vp
Cf2py intent(out):: d5
      !write(*,*) ' point_in_tetr--------------------------------'
      !write(*,*) v1
      !write(*,*) v2
      !write(*,*) v3
      !write(*,*) v4
      !write(*,*) 'vp=',vp
      
      call det4(v1,v2,v3,v4,d5(0))
      call det4(vp,v2,v3,v4,d5(1))
      call det4(v1,vp,v3,v4,d5(2))
      call det4(v1,v2,vp,v4,d5(3))
      call det4(v1,v2,v3,vp,d5(4))
      end subroutine point_in_tetr
c-------------------------------------------------------------------------
      subroutine numb_streams(q_lim,ngr,x0,x1,x2,x_lim,dgr,
     *                            nf0,nf1,nf2,nstream)
      IMPLICIT NONE
Cf2py intent(in):: q_lim
Cf2py intent(in):: ngr
Cf2py intent(in):: x0
Cf2py intent(in):: x1 
Cf2py intent(in):: x2
Cf2py intent(in):: x_lim
Cf2py intent(in):: dgr
Cf2py intent(in):: nf0
Cf2py intent(in):: nf1
Cf2py intent(in):: nf2
Cf2py intent(out):: nstream
      integer(8),dimension(0:5)::q_lim
      integer(8)::ngr
      real(8),dimension(0:ngr,0:ngr,0:ngr):: x0,x1,x2
      real(8),dimension(0:5):: x_lim      
      real(8),dimension(0:2)::dgr
      integer(8):: nf0,nf1,nf2
      integer(8),dimension(0:nf0-1,0:nf1-1,0:nf2-1):: nstream 
      
      real(8),   dimension(0:7,0:2):: vcube
      integer(8),dimension(0:4,0:3):: sv_e, sv_o, sv
      integer(8),dimension(0:3):: sv_r
      integer(8),dimension(0:2):: n_f_gr
      real(8),   dimension(0:3,0:2):: sv_s
      real(8),   dimension(0:2):: x_min,x_max
      integer(8),dimension(0:2):: ijk_min, ijk_max
      real(8),   dimension(0:4)::d5
      real(8),   dimension(0:2)::v1,v2,v3,v4,p
      integer(8)::i,j,k,i1,j1,k1,ns,ind_vert,ic,ind_cond,ig,jg,kg,par
c----Vertices in 5 Ss,starting from largest central oriented to make V>0 inL
      sv_e(0,:)= (/0,6,5,3/)  ! Even
      sv_e(1,:)= (/1,5,3,0/); sv_e(2,:)= (/2,0,3,6/)
      sv_e(3,:)= (/4,6,5,0/); sv_e(4,:)= (/7,3,5,6/)

      sv_o(0,:)= (/1,2,4,7/)   ! Odd
      sv_o(1,:)= (/0,1,2,4/); sv_o(2,:)= (/3,1,7,2/)
      sv_o(3,:)= (/6,2,7,4/); sv_o(4,:)= (/5,1,4,7/)
      
      n_f_gr(0)=nf0; n_f_gr(1)=nf1; n_f_gr(2)=nf2
      !write(*,*) 'n_f_gr=', n_f_gr
      do i = 0, nf0-1           ! set nstream to 0
         do j = 0 , nf1-1
            do k = 0, nf2-1
               nstream(i,j,k)=0
            enddo
         enddo
      enddo
      
      do i = q_lim(0), q_lim(1)-2 ! go over all particles in L-space   changed from 1 to 2 for large box
         do j = q_lim(2), q_lim(3)-2 ! including upper limit
            do k = q_lim(4), q_lim(5)-2
c.............. assign coordinates to cubicle vertices
               do i1 = 0, 1
                  do j1 = 0, 1
                     do k1 = 0, 1
                        vcube(0+4*k1+2*j1+i1,0) = x0(i+i1,j+j1,k+k1)
                        vcube(0+4*k1+2*j1+i1,1) = x1(i+i1,j+j1,k+k1)
                        vcube(0+4*k1+2*j1+i1,2) = x2(i+i1,j+j1,k+k1)
                     enddo
                  enddo
               enddo
               if (i < -1) then
                  write(*,*) '======================================='
                  write(*,*) 'i,j,k,', i,j,k
                  write(*,*) 'i,j,k,', i-q_lim(0),j-q_lim(2),k-q_lim(4)
                  !write(*,*) 'vcube'
                  !write(*,*) vcube(0,0:2)
                  !write(*,*) vcube(1,0:2)
                  !write(*,*) vcube(2,0:2)
                  !write(*,*) vcube(3,0:2)
                  !write(*,*) vcube(4,0:2)
                  !write(*,*) vcube(5,0:2)
                  !write(*,*) vcube(6,0:2)
                  !write(*,*) vcube(7,0:2)
                  !write(*,*) ' '
                  !read(*,*)
               endif
               
c.............. Choose the simplex reference list according to parity of cubicle
               par = modulo((i+j+k -q_lim(0)-q_lim(2)-q_lim(4)),2)
               if (par .eq. 1) then ! Odd
                  sv(:,:) = sv_o(0:4,0:3)
               else             ! Even
                  sv(:,:) = sv_e(0:4,0:3)
               endif
c.............. Loop over 5 simpleces in a cubicle
               do ns = 0, 4
                  !write(*,*) '------------- ns = ', ns, '----------'
                  sv_r(:) = sv(ns,0:3) ! simplex vertex reference
                  !write(6,*) 'sv_r', sv_r
                  !write(*,*) ' '
c................. for selected vertex copy coordinates 
                  do ind_vert = 0, 3 
                     sv_s(ind_vert,:)=vcube(sv_r(ind_vert),0:2) !simpl.vertices
                  enddo
                  if (i < -1)  then
                     write(*,*) 'sv_s'
                     write(*,*) sv_s(0,:)
                     write(*,*) sv_s(1,:)
                     write(*,*) sv_s(2,:)
                     write(*,*) sv_s(3,:)
                     write(*,*) ' '
                     !read(*,*)
                  endif
                  
c....find MIN and MAX of each vertex coordinates & min and max indeces in E
                  ind_cond = 1
                  do ic = 0, 2  ! index of coord i.e. 0,1,2
                     x_min(ic) = 
     *                 min(sv_s(0,ic),sv_s(1,ic),sv_s(2,ic),sv_s(3,ic))
                     ijk_min(ic)=int(floor((x_min(ic)-x_lim(2*ic))
     *                                                  /dgr(ic)))
                     if (ijk_min(ic) < 0) then
                        ijk_min(ic) = 0
                     endif
                     
                     x_max(ic) = 
     *                 max(sv_s(0,ic),sv_s(1,ic),sv_s(2,ic),sv_s(3,ic))
                     ijk_max(ic)=int(ceiling((x_max(ic)-x_lim(2*ic))
     *                                                    /dgr(ic)))
                     if (ijk_max(ic) > n_f_gr(ic)-1) then
                        ijk_max(ic) = n_f_gr(ic)-1
                     endif
                     
                     if (ijk_max(ic) < ijk_min(ic)) then 
                        ind_cond = 0
                     endif
                  enddo
                  if (i < -1) then
                     write(*,*) 'x_min', x_min
                     write(*,*) 'x-max', x_max
                     write(*,*) ' '
                     write(*,*) 'x_lim', x_lim
                     write(*,*) ' '
                     write(*,*) 'dgr', dgr
                     write(*,*) ' '

                     !write(*,*) 'i,j,k', i,j,k
                     write(*,*) 'ijk_min', ijk_min
                     write(*,*) 'ijk_max', ijk_max
                     write(*,*) ' '
                     write(*,*) 'ind_cond=', ind_cond
                     !read(*,*)
                  endif
c.................                  
                  if (ind_cond > 0) then
                     !write(*,*) 'ijk_min',ijk_min
                     !write(*,*) 'ijk_max',ijk_max
                     !read(*,*)
                     do ig = ijk_min(0), ijk_max(0)
                        p(0) = x_lim(0) + ig*dgr(0)
                        do jg = ijk_min(1), ijk_max(1)
                           p(1) = x_lim(2) + jg*dgr(1)
                           do kg = ijk_min(2), ijk_max(2)
                              p(2) = x_lim(4) + kg*dgr(2)
                              i1=i-q_lim(0)
                              j1 =j-q_lim(2)
                              k1 = k-q_lim(4)
                              v1=sv_s(0,:)
                              v2=sv_s(1,:)
                              v3=sv_s(2,:)
                              v4=sv_s(3,:)
                              
                              
                              call point_in_tetr(v1,v2,v3,v4,p,d5)
                              
                              if(i1 < -1000) then
                                 write(*,*) 'i1,j1,k1', i1,j1,k1
                                 write(*,*) 'ig,jg,kg', ig,jg,kg
                                 write(*,*) 'v1 ',v1
                                 write(*,*) 'v2 ',v2
                                 write(*,*) 'v3 ', v3
                                 write(*,*) 'v4 ', v4
                                 write(*,*) 'p ',  p    
                                 write(*,101) d5
                                 write(*,*) '---------------------'
                             	 !read(*,*)
                              endif
                              
                              if (i < -1) then
                                 write(*,*) 'sv_s(0,:)=',sv_s(0,:)
                                 write(*,*) sv_s(1,:)
                                 write(*,*) sv_s(2,:)
                                 write(*,*) sv_s(3,:)
                                 write(*,*) 'p=', p
                                 write(*,*) 'ig,jg,kg=', ig,jg,kg
                                 write(*,*) 'd5'
                                 write(*,101) d5
                                 write(*,*) '-----------------d5'
                                 read(*,*)
                              endif
                              if ((d5(0)*d5(1)>0) .and. (d5(0)*d5(2)>0) 
     *                             .and.(d5(0)*d5(3)>0)
     *                             .and.(d5(0)*d5(4)>0)) then
                                 nstream(ig,jg,kg)=nstream(ig,jg,kg)+1
                                 if (i > -1) then
                                    !write(*,*) '=================='
                                    
                                    !write(*,100) i1,j1,k1,ns, ig,jg,kg,
!     *                                        nstream(ig,jg,kg)
 !100  format(8i2)
 101                                format(5(f12.4))
                                    !write(*,*) 'ijk_min',ijk_min
                                    !write(*,*) 'ijk_max',ijk_max
                                    !write(*,*) ' '
                                    !write(*,*) 'sv_s(0,:)=',sv_s(0,:)
                                    !write(*,*) sv_s(1,:)
                                    !write(*,*) sv_s(2,:)
                                    !write(*,*) sv_s(3,:)
                                    !write(*,*) ' '
                                    !write(*,*) 'ig,jg,kg',ig,jg,kg 
                                    !write(*,*) 'p', p
                                    !write(*,*)nstream(ig,jg,kg)
                                    !write(*,*) ' '
                                    !write(*,*) '=================='
                                    !read(*,*)
                                 endif
                              endif
                           enddo
                        enddo
                     enddo
                  endif
                  !write(*,*) '-----------------------------------------'
                  !write(*,*) ' end ns loop'
                  !read(*,*)
               enddo            ! Loop over 5 simpleces
            enddo
         enddo
         !write(*,*) 'i', i
      enddo
      end subroutine numb_streams
c-------------------------------------------------------------------------
      subroutine count_streams(nstream,nf0,nf1,nf2, count,nc_max)
	  IMPLICIT NONE
c nstream = numb_streams(n1,x0,x1,x2,xegr,yegr,zegr,
c    ni=(shape(x0,0)-1),nj=(shape(x0,1)-1),nk=(shape(x0,2)-1),ngr=len(xegr))
Cf2py intent(in):: nstream
Cf2py intent(in):: nf0
Cf2py intent(in):: nf1
Cf2py intent(in):: nf2
Cf2py intent(in):: nc_max
Cf2py intent(out):: count
      integer(8):: nf0,nf1,nf2,nc_max
      integer(8),dimension(0:nf0-1,0:nf1-1,0:nf2-1):: nstream
      integer(8),dimension(0:nc_max-1):: count
      integer(8):: i,j,k,ns
      do i = 0, nc_max-1
         count(i) = 0
      enddo
      do i = 0, nf0-1
         do j = 0, nf1-1
            do k = 0, nf2-1
               ns = nstream(i,j,k)
               count(ns) = count(ns)+1
            enddo
         enddo
      enddo
      end subroutine count_streams
c--------------------------------------------------------------------------