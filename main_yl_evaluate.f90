program main_yl_evaluate
    use mod_yl, only: ik, rk, stderr
    use mod_yl_python, only: evaluate_yl
    implicit none

    character(len=:), allocatable :: input_file, output_file
    character(len=512) :: message
    integer :: path_length, ios, input_unit, output_unit
    integer(ik) :: nelem, nunknowns, i
    real(rk), allocatable :: u(:), residual(:), gradient(:)
    real(rk) :: bond 

    if (command_argument_count() /= 2) then
        error stop 'Usage: yl_evaluate input.dat output.dat'
    end if

    call get_command_argument(1, length=path_length)
    allocate(character(len=path_length) :: input_file)
    call get_command_argument(1, value=input_file)

    call get_command_argument(2, length=path_length)
    allocate(character(len=path_length) :: output_file)
    call get_command_argument(2, value=output_file)

    open(newunit=input_unit, file=input_file, status='old', &
         action='read', iostat=ios, iomsg=message)
    if (ios /= 0) then
        write(stderr,'(a)') trim(message)
        error stop 'Cannot open input'
    end if

    read(input_unit,*,iostat=ios) nelem, bond 
    if (ios /= 0) error stop 'Cannot read nelem and bond'
    if (nelem < 1) error stop 'nelem must be positive'
    if (nelem > (huge(nelem)-3)/4) error stop 'nelem too large'

    nunknowns = 4*nelem + 3
    allocate(u(nunknowns))

    do i = 1, nunknowns
        read(input_unit,*,iostat=ios) u(i)
        if (ios /= 0) error stop 'Cannot read complete U vector'
    end do
    close(input_unit)

    call evaluate_yl(nelem, bond, u, residual, gradient)

    open(newunit=output_unit, file=output_file, status='replace', &
         action='write', iostat=ios, iomsg=message)
    if (ios /= 0) then
        write(stderr,'(a)') trim(message)
        error stop 'Cannot open output'
    end if

    do i = 1, nunknowns
        write(output_unit,'(2es26.17e3)',iostat=ios) &
            residual(i), gradient(i)
        if (ios /= 0) error stop 'Cannot write output'
    end do

    close(output_unit, iostat=ios)
    if (ios /= 0) error stop 'Cannot close output'
end program main_yl_evaluate