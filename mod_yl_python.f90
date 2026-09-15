module mod_yl_python
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite

    use mod_yl, only: ik, rk, yl_governers, alloc, &
        amat, rowmat, colmat, head, rhs, yl_soln, &
        myband, yl_gp, yl_ph1, yl_dph1, yl_phi1d, &
        fill_in_rhs_amat_projection, &
        fill_in_colmat_projection, &
        fill_in_rowmat_rhscon

    use mod_yl_benchmark, only: setup_benchmark

    implicit none
    private
    public :: evaluate_yl

contains

    subroutine evaluate_yl(nelem, bond, u, residual, gradient)
        integer(ik), intent(in) :: nelem
        real(rk), intent(in) :: bond
        real(rk), intent(in) :: u(:)
        real(rk), allocatable, intent(out) :: residual(:)
        real(rk), allocatable, intent(out) :: gradient(:)

        integer(ik), parameter :: bw = 13
        integer(ik) :: nnodes, ncoords, nunknowns
        integer(ik) :: elem, k, first, i, j, band_col, mid

        type(yl_governers) :: problem
        real(rk) :: length
        real(rk) :: dr, dz, tangent_squared

        ! All problem metadata is local and assigned on every call.
        if (nelem < 1) error stop 'nelem must be positive'

        nnodes = 2*nelem + 1
        ncoords = 2*nnodes
        nunknowns = ncoords + 1

        if (size(u) /= nunknowns) then
            error stop 'Incorrect unknown-vector size'
        end if
        if (.not. all(ieee_is_finite(u))) then
            error stop 'Input U contains NaN or Inf'
        end if

       if (.not. ieee_is_finite(bond)) then
            error stop 'Nonfinite bond'
        end if

        call setup_benchmark(length, problem, bond)

        ! The pressure used by assembly must come from the NN.
        problem%p0 = u(nunknowns)

        ! Existing module work arrays: reset on every evaluation.
        myband = bw

        call alloc(amat,   n=[ncoords, (3*bw-1)/2])
        call alloc(rowmat,n=[1, ncoords])
        call alloc(colmat,n=[ncoords, 1])
        call alloc(head,  n=[1, 1])
        call alloc(rhs, nunknowns)
        call alloc(yl_soln, nunknowns)

        call yl_phi1d(yl_gp, yl_ph1, yl_dph1)

        yl_soln = u
        amat   = 0.0_rk
        rowmat = 0.0_rk
        colmat = 0.0_rk
        head   = 0.0_rk
        rhs    = 0.0_rk

        ! Reject undefined geometry before dividing by tangent length.
        do elem = 1, nelem
            first = 4*(elem-1) + 1
            do k = 1, 3
                dr = dot_product(yl_dph1(:,k), &
                                 u(first:first+4:2))
                dz = dot_product(yl_dph1(:,k), &
                                 u(first+1:first+5:2))
                tangent_squared = dr*dr + dz*dz

                if (.not. ieee_is_finite(tangent_squared)) then
                    error stop 'Nonfinite tangent'
                end if
                if (tangent_squared <= tiny(1.0_rk)) then
                    error stop 'Zero tangent length'
                end if
            end do
        end do

        ! Existing physics assembly, unchanged.
        do elem = 1, nelem
            call fill_in_rhs_amat_projection(elem, problem)
            call fill_in_colmat_projection(elem)
            call fill_in_rowmat_rhscon(elem, problem)
        end do

        ! Existing boundary-condition assembly, unchanged.
        call problem%bc_b0r(eqnnum=1,         bdelem=1)
        call problem%bc_b0z(eqnnum=2,         bdelem=1)
        call problem%bc_bfr(eqnnum=ncoords-1, bdelem=nelem)
        call problem%bc_bfz(eqnnum=ncoords,   bdelem=nelem)

        if (.not. all(ieee_is_finite(rhs))) then
            error stop 'Nonfinite residual'
        end if
        if (.not. all(ieee_is_finite(amat))) then
            error stop 'Nonfinite band Jacobian'
        end if
        if (.not. all(ieee_is_finite(rowmat))) then
            error stop 'Nonfinite Jacobian row'
        end if
        if (.not. all(ieee_is_finite(colmat))) then
            error stop 'Nonfinite Jacobian column'
        end if
        if (.not. all(ieee_is_finite(head))) then
            error stop 'Nonfinite Jacobian head'
        end if

        allocate(residual(nunknowns), gradient(nunknowns))

        ! rhs is R here, not the negated Newton RHS.
        residual = rhs
        gradient = 0.0_rk

        ! Compute gradient = J^T R using existing band/arrow storage.
        mid = (bw+1)/2

        do i = 1, ncoords
            do band_col = 1, bw
                j = i + band_col - mid
                if (j >= 1 .and. j <= ncoords) then
                    gradient(j) = gradient(j) &
                        + amat(i,band_col)*residual(i)
                end if
            end do
        end do

        gradient(1:ncoords) = gradient(1:ncoords) &
            + rowmat(1,1:ncoords)*residual(nunknowns)

        gradient(nunknowns) = &
            dot_product(colmat(1:ncoords,1),residual(1:ncoords)) &
            + head(1,1)*residual(nunknowns)

        if (.not. all(ieee_is_finite(gradient))) then
            error stop 'Nonfinite loss gradient'
        end if

        ! No rhs=-rhs.
        ! No ArrowSolver.
        ! No Newton update.
    end subroutine evaluate_yl

end module mod_yl_python