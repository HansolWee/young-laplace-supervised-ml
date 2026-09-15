module mod_yl_benchmark
    use mod_yl, only: ik, rk, pi, yl_governers, spline_objective, &
                      make_s_from_xy, make_2nd_order_deriv, alloc 
    implicit none
    private
    public :: setup_benchmark, build_initial_guess

contains

    subroutine setup_benchmark(L, yl, bond)
        real(rk), intent(out) :: L
        type(yl_governers), intent(out) :: yl
        real(rk), intent(in) :: bond

        L = 2.0_rk

        yl%b0_r = 'essential'
        yl%b0_z = 'essential'
        yl%bf_r = 'essential'
        yl%bf_z = 'essential'

        yl%r0 = 1.0_rk
        yl%z0 = 0.0_rk
        yl%rf = 1.0_rk
        yl%zf = L

        yl%bond = bond
        yl%sigma = 1.0_rk
        yl%ref_i = 0.0_rk
        yl%v0 = pi*L

        ! Initial pressure guess
        yl%p0 = yl%sigma - 0.5_rk*bond*L
    end subroutine setup_benchmark

    subroutine build_initial_guess(n, L, eps, curve)
        integer(ik), intent(in) :: n
        real(rk), intent(in) :: L, eps
        type(spline_objective), intent(inout) :: curve

        integer(ik) :: i
        real(rk) :: z

        call alloc(curve%xd, n)
        call alloc(curve%yd, n)
        call alloc(curve%sd, n)
        call alloc(curve%sx2d, n)
        call alloc(curve%sy2d, n)

        do i = 1, n
            z = L*real(i-1,rk)/real(n-1,rk)
            curve%yd(i) = z
            curve%xd(i) = 1.0_rk + eps*sin(pi*z/L)
        end do

        ! Make the pinned contact lines exact in the starting geometry.
        curve%xd(1) = 1.0_rk
        curve%yd(1) = 0.0_rk
        curve%xd(n) = 1.0_rk
        curve%yd(n) = L

        call make_s_from_xy(curve%xd,curve%yd,curve%sd)
        call make_2nd_order_deriv(curve%sd,curve%xd,2.0e30_rk,2.0e30_rk,curve%sx2d)
        call make_2nd_order_deriv(curve%sd,curve%yd,2.0e30_rk,2.0e30_rk,curve%sy2d)

    end subroutine build_initial_guess

end module mod_yl_benchmark
