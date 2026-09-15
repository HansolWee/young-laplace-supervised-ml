program main_liquid_bridge_benchmark
    use mod_yl, only: ik, rk, pi, yl_governers, spline_objective
    use mod_yl_benchmark, only: setup_benchmark, build_initial_guess
    implicit none

    ! ---------------------------------------------------------------------------
    ! Lowry & Steen (1995) fixed-contact-line liquid bridge benchmark.
    !
    ! The end discs have nondimensional radius 1, hence
    !   r(0) = r(S) = 1,
    !   z(0) = 0,
    !   z(S) = L.
    !
    ! The current mod_yl volume residual is
    !   Rv = pi * integral(r^2 dz) - v0,
    ! so v0 below uses that convention directly.
    ! ---------------------------------------------------------------------------

    integer(ik),      parameter :: ninit = 201_ik
    real(rk),         parameter :: perturbation = 1.0e-2_rk

    type(yl_governers)   :: yl
    type(spline_objective) :: initial_rz

    real(rk) :: L, volume_num
    integer(ik) :: nelem, i
    integer(ik) :: iu, ios
    character(len=64) :: argument
    real(rk) :: bond 

    nelem = 100

    if (command_argument_count() /= 2) then
        error stop 'Usage: yl nelem bond'
    end if

    call get_command_argument(1, argument)
    read(argument,*,iostat=ios) nelem
    if (ios /= 0) error stop 'Invalid nelem'
    if (nelem < 1) error stop 'nelem must be positive'

    call get_command_argument(2, argument)
    read(argument,*,iostat=ios) bond
    if (ios /= 0) error stop 'Invalid bond'

    call setup_benchmark(L, yl, bond)
    call build_initial_guess(ninit, L, perturbation, initial_rz)

    write(*,'(a)') repeat('=',78)
    write(*,'(a)') 'Lowry-Steen liquid-bridge benchmark'
    write(*,'(a,es16.8)') 'L                 : ', L
    write(*,'(a,es16.8)') 'Bond number       : ', yl%bond
    write(*,'(a,es16.8)') 'surface tension   : ', yl%sigma
    write(*,'(a,es16.8)') 'target v0         : ', yl%v0
    write(*,'(a,es16.8)') 'initial p0 guess  : ', yl%p0
    write(*,'(a)') repeat('=',78)

    call yl%solv(initial_rz,nelem_in=nelem)
    
    open(newunit=iu, file='u_newton.dat', status='replace', action='write')

    do i = 1, size(initial_rz%xd)
        write(iu,'(es26.17e3)') initial_rz%xd(i)
        write(iu,'(es26.17e3)') initial_rz%yd(i)
    end do
    write(iu,'(es26.17e3)') yl%p0
    
    close(iu)

    ! Diagnostic trapezoidal volume; not the FEM volume residual.
    volume_num = 0.0_rk
    do i = 1, size(initial_rz%xd)-1
        volume_num = volume_num + 0.5_rk*pi * &
            (initial_rz%xd(i)**2 + initial_rz%xd(i+1)**2) * &
            (initial_rz%yd(i+1)-initial_rz%yd(i))
    end do

    write(*,'(a)') repeat('=',78)
    write(*,'(a)') 'Benchmark summary'
    write(*,'(a,es16.8)') 'computed p0       : ', yl%p0
    write(*,'(a,es16.8)') 'computed volume   : ', volume_num
    write(*,'(a,es16.8)') '|V-V_target|      : ', abs(volume_num-yl%v0)
    write(*,'(a)') 'solution vector written to u_newton.dat'
    write(*,'(a)') repeat('=',78)

end program main_liquid_bridge_benchmark
