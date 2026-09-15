module mod_yl
    use iso_fortran_env, only: int32, stdout => output_unit, &
                                   stderr => error_unit
    implicit none
    ! ==============================================================================
    ! PRECISION KINDS DEFINITION
    ! ==============================================================================
    integer(int32), parameter :: rk  = selected_real_kind(p=15) ! Standard real kind (Double Precision, ~15 digits)
    integer(int32), parameter :: ik  = 4                        ! Standard integer kind (4 bytes)
    ! ==============================================================================
    ! GLOBAL CONSTANTS & VARIABLES
    ! ==============================================================================
    real(rk), parameter :: pi = 4.0_rk * ATAN(1.0_rk)           ! Mathematical constant Pi

    type spline_objective
        real(rk), allocatable :: xd(:), yd(:), sd(:), sx2d(:), sy2d(:)
    end type spline_objective

    interface alloc
        module procedure alloc_r1, alloc_r2
    end interface alloc
    
   !      2 (phase)
   !_____                          ↓ 𝐠 = -g e_z 
   !     \⩘ n (unit normal)
   !  1   \______
   !(phase)
   ! fluid statics: dp/dz = -ρg in both phase 1 and 2 
   ! solver for pressure in each phase, we then get  
   ! p₁ = C₁ - ρ₁gz --(1) 
   ! p₂ = C₂ - ρ₂gz --(2)  
   ! At free surface, p₂-p₁=2𝐇σ ⇒ Δp = p₂-p₁ = 2𝐇σ 
   ! define ΔC to be C₂-C₁      
   ! p₁-p₂ = C₁-C₂ + (ρ₂-ρ₁)gz ⇒ -Δp = -ΔC + Δρ⋅gz = -2𝐇σ
   ! we obtain dimensional YL equation: -ΔC + Δρ⋅gz = -2𝐇σ 
   ! let's make this equation dimensionless
   ! characteristic length scale = R 
   ! characteristic surface tension scale = σ₀ 
   ! dimensionless curvature 2ℍ = 2𝐇R 
   ! dimensionless coordinate z̃ = z/R 
   ! dimensionless surface tension σ̃ = σ/σ₀    
   ! ⇒ -ΔC + Δρ⋅gR z/R = -2𝐇R σ₀/R σ̃ 
   ! ⇒ -ΔC + Δρ⋅gR z̃ = -2ℍ σ₀/R σ̃ 
   ! ⇒ -ΔC R/σ₀ + Δρ⋅gR²/σ₀ z̃ = -2ℍσ̃
   ! let -ΔC R/σ₀ = P₀ and Δρ⋅gR²/σ₀ = G 
   ! we obtain dimensionless YL equation: P₀ + Gz̃ = -2ℍσ̃
   ! Consistent check; eqn (19) in Liao et. al. 2006: K − Gz = σ̃(∇ₛ⋅𝐧)
   ! In his definition, ρ₂=0 and G=ρ₁gR²/σ₀, i.e. exactly the same equation as described above 

   ! In what follows, all variables are dimensionless 
   
   ! notice that G<0 if phase 1 = liquid, phase 2 = gas
   ! notice that G>0 if phase 1 = gas, phase 2 = liquid 
   ! P₀: reference pressure will be determined via volume constraint 

   ! following is assumed (z ⇄ 𝐭 and r ⇄ 𝐧) 
   ! 𝐭 = rₛ e_r + zₛ e_z 
   ! 𝐧 = 𝛉 × 𝐭 = 𝛉 × (rₛ e_r + zₛ e_z) = zₛ e_r - rₛ e_z 
   ! 𝐭 × 𝐧 = (rₛ e_r + zₛ e_z) × (zₛ e_r - rₛ e_z) = (rₛ²+zₛ²) 𝛉 = 𝛉 
   
   ! then, -2ℍ = ∇ₛ⋅𝐧 = (𝐭 ∂/∂s + 𝛉/r ∂/∂θ)⋅(zₛ e_r - rₛ e_z) = zₛₛrₛ - rₛₛzₛ + zₛ/r 

   ! basis function ϕⁱ at i th node with ω(s) = 𝐧⋅e_r = zₛ
   ! derivation of GFEM residual form 
   ! Rᵢ = 2π ∫(-2ℍσ̃ - P₀ - Gz̃) ϕⁱ ω(s) r ds   with some weight function ω(s) 
   ! ⇒ Rᵢ =  ∫∫(-2ℍσ̃ - P₀ - Gz̃) ϕⁱ ω(s) dA
   ! ⇒ Rᵢ =  ∫∫-2ℍσ̃ ϕⁱω(s) dA - ∫∫ (P₀ + Gz̃) ϕⁱ ω(s) dA 
   ! ∫∫ ∇ₛ⋅𝐩 + 2ℍ𝐧⋅𝐩 dA = ∫ 𝐦⋅𝐩 dC, where 𝐦 is unit vector tangent to the surface and outwardly normal to the contour C  
   ! Thus, with 𝐩 = σ̃ϕⁱe_r, ∫∫-2ℍσ̃ ϕⁱω(s) dA = ∫∫-2ℍ𝐧⋅(σ̃ϕⁱe_r) dA = ∫∫ ∇ₛ⋅(σ̃ϕⁱe_r) dA - ∫ 𝐦⋅(σ̃ϕⁱe_r) dC
   ! ⇒ Rᵢ =  ∫∫ ∇ₛ⋅(σ̃ϕⁱe_r) dA - ∫ 𝐦⋅(σ̃ϕⁱe_r) dC - ∫∫ (P₀ + Gz̃) ϕⁱ (𝐧⋅e_r) dA
   ! ⇒ Rᵢ =  ∫∫ ∇ₛ⋅(σ̃ϕⁱe_r) dA - ∫ 𝐦⋅(σ̃ϕⁱe_r) dC - ∫∫ (P₀ + Gz̃) ϕⁱzₛ dA
   ! ∇ₛ⋅(σ̃ϕⁱe_r) = (𝐭 ∂/∂s + 𝛉/r ∂/∂θ)⋅(σ̃ϕⁱe_r) = rₛ d/ds(σ̃ϕⁱ) + σ̃ϕⁱ/r 
   ! ⇒ Rᵢ =  ∫∫ rₛ d/ds(σ̃ϕⁱ) + σ̃ϕⁱ/r  dA - ∫ 𝐦⋅(σ̃ϕⁱe_r) dC - ∫∫ (P₀ + Gz̃) ϕⁱzₛ dA 
   ! drop boundary terms and the factor 2π  
   ! ⇒ Rᵢ =  ∫ [σ̃rₛ d/ds(ϕⁱ) + σ̃ϕⁱ/r - (P₀ + Gz̃) ϕⁱzₛ] r ds  
   ! ⇒ Rᵢ =  ∫ [σ̃rₛr d/ds(ϕⁱ) + σ̃ϕⁱ - (P₀ + Gz̃)r ϕⁱzₛ] ds  
   ! alternative derivation 
   ! -2ℍ𝐧⋅(σ̃ϕⁱe_r) = σ̃ϕⁱzₛ(zₛₛrₛ - rₛₛzₛ + zₛ/r) = σ̃ϕⁱ(zₛₛzₛrₛ - rₛₛzₛ² + zₛ²/r)
   ! since rₛ²+zₛ²=1, rₛrₛₛ + zₛzₛₛ = 0. Thus, zₛzₛₛ = - rₛrₛₛ
   ! -2ℍ𝐧⋅(σ̃ϕⁱe_r) = σ̃ϕⁱ(zₛₛzₛrₛ - rₛₛzₛ² + zₛ²/r) = σ̃ϕⁱ(-rₛₛrₛ² - rₛₛzₛ² + zₛ²/r) = σ̃ϕⁱ(-rₛₛ + (1-rₛ²)/r)
   !               = σ̃ϕⁱ(-rₛₛ - rₛ²/r + 1/r) = σ̃ϕⁱ(-rₛₛ - rₛ²/r + 1/r) = σ̃ϕⁱ/r + σ̃ϕⁱ(-rₛₛ - rₛ²/r)
   !               = σ̃ϕⁱ/r - σ̃ϕⁱ/r d/ds(rrₛ) = σ̃ϕⁱ/r - σ̃ϕⁱ/r d/ds(rrₛ) = σ̃ϕⁱ/r + rₛ d/ds(σ̃ϕⁱ) - 1/r d/ds(σ̃ϕⁱrrₛ)
   ! d/ds(σ̃ϕⁱrrₛ) = σ̃ϕⁱd/ds(rrₛ) + rrₛ d/ds(σ̃ϕⁱ) ⇒ -σ̃ϕⁱ/r d/ds(rrₛ) = rₛ d/ds(σ̃ϕⁱ) - 1/r d/ds(σ̃ϕⁱrrₛ)
   ! Rᵢ = ∫(-2ℍσ̃ - P₀ - Gz̃) ϕⁱ ω(s) r ds
   ! ⇒ Rᵢ = ∫(-2ℍσ̃)ϕⁱ zₛ r - (P₀ + Gz̃) ϕⁱ zₛ r ds
   ! ⇒ Rᵢ = ∫ σ̃ϕⁱ + rrₛ d/ds(σ̃ϕⁱ) - d/ds(σ̃ϕⁱrrₛ) - (P₀ + Gz̃) ϕⁱ zₛ r ds
   ! ⇒ Rᵢ = ∫ σ̃ϕⁱ + σ̃rrₛ d/ds(ϕⁱ) - (P₀ + Gz̃) ϕⁱ zₛ r ds - (σ̃ϕⁱrrₛ)|boundary term
   !______________________________________________________________________________________!
   !                                                                                      !
   !                    Rᵢ =  ∫ [σ̃rₛr d/ds(ϕⁱ) + σ̃ϕⁱ - (P₀ + Gz̃)r ϕⁱzₛ] ds                 !
   !______________________________________________________________________________________! 
   ! derivation of GFEM form of volume constraint 
   ! Rᵥ = V - V₀ 
   ! ⇒ Rᵥ = π ∫ r² dz - V₀
   ! ⇒ Rᵥ = π ∫ r²zₛ ds - V₀   
   !______________________________________________________________________________________!
   !                                                                                      !
   !                                  Rᵥ = π∫ r²zₛ ds -  V₀                               !
   !______________________________________________________________________________________!  
 
   type yl_governers
      real(rk) :: bond    ! G 
      real(rk) :: sigma   ! σ̃
      real(rk) :: p0      ! P₀ = C₁ - C₂ (dimensionless)
      real(rk) :: ref_i   ! either C₁ or C₂; C₁ = P₀ + C₂ and C₂ = C₁ - P₀     
      real(rk) :: v0      ! V₀; P₀ is determined by volume constraint 
      character(len=:), allocatable :: b0_r, b0_z, bf_r, bf_z  
      real(rk) :: r0, z0  ! boundary conditions at the starting point of integration 
      real(rk) :: rf, zf  ! boundary conditions at the end point of integration 
   contains 
      procedure, pass(this), public :: solv => young_laplacer
      procedure, pass(this), public :: bc_b0r
      procedure, pass(this), public :: bc_b0z
      procedure, pass(this), public :: bc_bfr
      procedure, pass(this), public :: bc_bfz  
   end type yl_governers

   ! we then consider computational coordinate χ 
   ! consider the map between arclength 's' and computational coordinate χ 
   ! this mapping is invertible, i.e. s=s(χ) and χ=χ(s)  
   ! we label the nodes by assigning an integer tag to each of them
   ! at the first node, χ₁ = 0 
   ! at the second node, χ₂ = 2 
   ! at the third node, χ₃ = 4 
   ! ...
   ! at the last node, χ = 2*(ylnodes - 1) (ylnodes: total node numbers involved in computation)
   ! Thus, at each node, χ is always constant. 
   ! this does not hold for arclength because while Newton Raphson iteration is going on 
   ! the positions of nodes are adjusted and hence the values of arclength at each nodal point also change   
   ! differential of arclength can be easily obtained as ds = (rᵪ²+zᵪ²)^(1/2) dχ 
   ! for notational simplicity, let (rᵪ²+zᵪ²)^(1/2) = sᵪ   
   ! recast the equations in terms of χ 
   
   ! Rᵢ =  ∫ [(σ̃zᵪ/(rsᵪ) - P₀ - Gz̃) ϕⁱsᵪ²/zᵪ² + σ̃ϕᵪⁱrᵪ/(sᵪzᵪ)] sᵪdχ
   ! ⇒ Rᵢ =  ∫ (σ̃zᵪ/r - P₀sᵪ - Gz̃sᵪ) ϕⁱsᵪ²/zᵪ² + σ̃ϕᵪⁱrᵪ/zᵪ dχ
   !______________________________________________________________________________________!
   !                                                                                      !
   !                Rᵢ =  ∫ (σ̃zᵪ/r - P₀sᵪ - Gz̃sᵪ) ϕⁱsᵪ²/zᵪ² + σ̃ϕᵪⁱrᵪ/zᵪ dχ                ! 
   !                                                                                      !
   !                              Rᵥ = π∫ r²zᵪ dχ -  V₀                                   !
   !______________________________________________________________________________________!
   ! or we use the other formulation 
   ! Rᵢ =  ∫ [σ̃rₛr d/ds(ϕⁱ) + σ̃ϕⁱ - (P₀ + Gz̃)r ϕⁱzₛ] ds                 
   ! ⇒ Rᵢ =  ∫ [σ̃rᵪ/sᵪr d/dχ(ϕⁱ)/sᵪ + σ̃ϕⁱ - (P₀ + Gz̃)r ϕⁱzᵪ/sᵪ] sᵪdχ                 
   ! ⇒ Rᵢ =  ∫ [σ̃rrᵪ/sᵪ d/dχ(ϕⁱ) + σ̃ϕⁱsᵪ - (P₀ + Gz̃)r ϕⁱzᵪ] dχ        
   !______________________________________________________________________________________!
   !                                                                                      !
   !              Rᵢ =  ∫ [σ̃rrᵪ/sᵪ d/dχ(ϕⁱ) + σ̃ϕⁱsᵪ - (P₀ + Gz̃)r ϕⁱzᵪ] dχ                 !
   !                                                                                      !
   !                              Rᵥ = π∫ r²zᵪ dχ -  V₀                                   !
   !______________________________________________________________________________________!         
   ! 
   ! since there are two variables at each node (r,z) 
   ! a node distribution function is also solved in addition to YL 
   !______________________________________________________________________________________!
   !                                                                                      !
   !                            Rᵢ =  ∫ (rᵪ² + zᵪ²)^(1/2) ϕᵪⁱ dχ                          !
   !                                                                                      !
   !______________________________________________________________________________________!
   
   ! arrays used in arrowsolver 
   real(rk), allocatable :: rhs(:)
   real(rk), allocatable :: head(:,:) 
   real(rk), allocatable :: colmat(:,:)
   real(rk), allocatable :: rowmat(:,:)
   real(rk), allocatable :: amat(:,:)
   !       RHS(NOD+IARROW) -       HOLDS B VECTORS.  THE SOLUTION          | 
   !                               VECTOR X IS RETURNED IN THIS VECTOR     | 
   !       HEAD(IARROW,IARROW) -   HOLDS THE ELEMENTS IN THE SQUARE        | 
   !                               SECTION OF THE ARROW MATRIX             | 
   !       ROWMAT(IARROW,NOD) -    HOLDS THE ELEMENTS OF THE ROWS          | 
   !                               OF THE ARROW MATRIX                     | 
   !       COLMAT(NOD,IARROW) -    HOLDS THE ELEMENTS OF THE COLUMNS       | 
   !                               OF THE ARROW MATRIX                     | 
   !       AMAT(NOD, 3*IBAND-1 ) - BANDED MATRIX A.  NOTE THAT THE         | 
   !                 ---------     DIMENSIONS ARE 50% GREATER THAN         | 
   !                     2         THE BANDWIDTH.                          | 

   ! band width (used by bgbl)
   integer(ik) :: myband 

   ! solution array 
   real(rk), allocatable :: yl_soln(:)

   ! gauss points, weights and 1d quadratic basis functions 
   real(rk), parameter :: yl_wt(3) = [(0.555555555555556_rk)*0.5_rk, &
                                    (0.888888888888889_rk)*0.5_rk,&
                                    (0.555555555555556_rk)*0.5_rk]
   real(rk), parameter :: yl_gp(3) = [(1.0_rk-0.774596669241483_rk)*0.5_rk, &
                                    0.5_rk,(1.0_rk+0.774596669241483_rk)*0.5_rk] 
   real(rk), allocatable :: yl_ph1(:,:), yl_dph1(:,:)

    ! after YL is solved, pressure field can be calculated with the aid of the following derived type 
    type :: hydrostatic 
        type(yl_governers) :: yl 
    contains
        procedure, pass(params) :: comp => pressure_field
    end type hydrostatic
    contains

    subroutine alloc_r1(a,n)
       real(rk), allocatable, intent(inout) :: a(:)
       integer(ik), intent(in) :: n

       if (allocated(a)) deallocate(a)
       allocate(a(n))
       a = 0.0_rk
    end subroutine alloc_r1

    subroutine alloc_r2(a,n)
       real(rk), allocatable, intent(inout) :: a(:,:)
       integer(ik), intent(in) :: n(2)

       if (allocated(a)) deallocate(a)
       allocate(a(n(1),n(2)))
       a = 0.0_rk
    end subroutine alloc_r2

   pure elemental real(rk) function pressure_field( x, params )
      real(rk), intent(in) :: x
      class(hydrostatic), intent(in) :: params
      ! definitions 
      ! -ΔC R/σ₀ = P₀ and Δρ⋅gR²/σ₀ = G
      ! C̃₁ = P₀ + C̃₂ and C̃₂ = C̃₁ - P₀ 
      !______________________________________________________________________________________!
      ! case 1 (phase 1 = liquid and phase 2 = gas) 
      ! ρ₂ = 0 and C₂ = 0 ⇐ pressure of gas is set to be 0 (pressure datum) 
      ! dimensionless C₁ is then given as C₁ = P₀ + C₂ = P₀  
      ! p₁ = C₁ - ρ₁gz ⇒ p̃₁ = C̃₁ - ρ₁gR²/σ₀ z̃ ⇒ p̃₁ = C̃₁ + Gz (G = -ρ₁gR²/σ₀ < 0)
      ! <<<<<<<<<<<<<<<<<<<<<<<<<<<<when G<0, p̃₁ = P₀ + Gz>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>      
      !______________________________________________________________________________________!
      ! case 2 (phase 1 = gas and phase 2 = liquid)
      ! ρ₁ = 0 and C₁ = params%yl%ref_i ⇐ pressure of gas is initially set to be some constant  
      ! dimensionless C̃₂ is then given as C̃₂ = C̃₁ - P₀ = params%yl%ref_i - P₀  
      ! p₂ = C₂ - ρ₂gz ⇒ p̃₂ = C̃₂ - ρ₂gR²/σ₀ z̃ = C̃₁ - P₀ - Gz̃ (G = ρ₂gR²/σ₀ > 0) 
      ! <<<<<<<<<<<<<<<<<<<<<<<<<<<<when G>0, p̃₂ = C̃₁ - P₀ - Gz̃>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      !______________________________________________________________________________________!

      if (params%yl%bond<0.0) then 
         pressure_field = params%yl%p0 + params%yl%bond * x 
      else
         pressure_field = params%yl%ref_i - params%yl%p0 - params%yl%bond * x 
      end if 
      
   end function pressure_field
    
    subroutine young_laplacer(this,initial_rz,nelem_in)
        class(yl_governers) :: this 
        type(spline_objective) :: initial_rz  
        integer(ik), intent(in), optional :: nelem_in
        integer(ik), parameter :: ibw=13
        integer(ik), parameter :: iaw=1  
        integer(ik) :: ylnodes, ylelems, ylnunks
        integer(ik) :: dof_l      
        integer(ik) :: iteration 
        real(rk) :: toterr 
        integer(ik) :: n  
        
      
        ! set up arrow solver
        dof_l = 2   
        ylelems = 20000
        if (present(nelem_in)) ylelems = nelem_in
        if (ylelems < 1) error stop 'nelem_in must be positive'
      ylnodes = 2*ylelems + 1   
      ylnunks = dof_l*ylnodes + iaw 
      call alloc(amat,n=[dof_l*ylnodes,(3*ibw-1)/2])
      call alloc(rowmat,n=[iaw,dof_l*ylnodes])
      call alloc(colmat,n=[dof_l*ylnodes,iaw])
      call alloc(head,n=[iaw,iaw])
      call alloc(rhs,ylnunks)
      
      call alloc(yl_soln,ylnunks)

      ! set global bandwidth (myband is module variable)
      myband = ibw 

      ! set up 1d quadratic basis functions
      ! yl_gp, yl_ph1, yl_dph1 are module variables (defined before contains)
      call yl_phi1d(yl_gp,yl_ph1,yl_dph1)

      ! initial conditions for (r,z) from initial_rz 
      call obtain_initial_condition(yl_soln,ylnodes,initial_rz)
      ! initial condition for reference pressure 
      yl_soln(size(yl_soln)) = this%p0 

      ! initial condition checker 
      call meshwriter_1d('yl.dat',1,initial_rz%xd,initial_rz%yd)

      ! reallocate initial_rz 
      call alloc(initial_rz%xd,ylnodes)
      call alloc(initial_rz%yd,ylnodes)
      call alloc(initial_rz%sd,ylnodes)
      call alloc(initial_rz%sx2d,ylnodes)
      call alloc(initial_rz%sy2d,ylnodes) 
      
      print '(a)', repeat('=', 128)
      iteration = 0 
      do
         iteration = iteration + 1 
         
         amat = 0.0_rk; rowmat = 0.0_rk; colmat = 0.0_rk; head = 0.0_rk; rhs = 0.0_rk 
         ! yl assembly 
         do n=1,ylelems 
            call fill_in_rhs_amat_projection(n,this)
            call fill_in_colmat_projection(n) 
            call fill_in_rowmat_rhscon(n,this)
            ! do not need to compute head because Rᵥ = π∫ r²zᵪ dχ -  V₀ ,i.e. it lacks of P₀ 
         end do  
         
         ! impose bc  
         call this%bc_b0r(eqnnum=1,bdelem=1)
         call this%bc_b0z(eqnnum=2,bdelem=1)
         call this%bc_bfr(eqnnum=2*ylnodes-1,bdelem=ylelems)
         call this%bc_bfz(eqnnum=2*ylnodes+0,bdelem=ylelems)
         
         ! call arrow solver 
         rhs = - rhs
         call ArrowSolver(2*ylnodes,ibw,iaw,0.25_rk)
          
         ! update solution 
         yl_soln = yl_soln + rhs
         this%p0 = yl_soln(size(yl_soln))

         ! solution 
         call dessembler(yl_soln,ylnodes,initial_rz%xd,initial_rz%yd)
         call meshwriter_1d('yl.dat',iteration+1,initial_rz%xd,initial_rz%yd)

         ! error
         call l2_norm(iteration,ylnodes,rhs,toterr)
 
         ! exit condition 
         if (toterr<1.0E-6_rk) exit
      end do 
      call make_s_from_xy(initial_rz%xd,initial_rz%yd,initial_rz%sd)
      call make_2nd_order_deriv(initial_rz%sd,initial_rz%xd,2.0E30_rk,2.0E30_rk,initial_rz%sx2d)
      call make_2nd_order_deriv(initial_rz%sd,initial_rz%yd,2.0E30_rk,2.0E30_rk,initial_rz%sy2d)
      print '(a)', repeat('=', 128)
      call meshwriter_1d('yl.dat',1,initial_rz%xd,initial_rz%yd)
      
   end subroutine young_laplacer 
   subroutine bc_b0r(this,eqnnum, bdelem)
      implicit none 
      class(yl_governers) :: this 
      integer(ik), intent(in) :: eqnnum, bdelem 
      select case(this%b0_r)
      case('essential')
         amat(eqnnum,:) = 0.0_rk; colmat(eqnnum,1) = 0.0_rk; rhs(eqnnum) = 0.0_rk 
         amat(eqnnum,bgbl(2,bdelem,1,1,0,0)) = 1.0_rk; 
         rhs(1) = yl_soln(1) - this%r0
      case('switch')
         amat(eqnnum,:) = 0.0_rk; colmat(eqnnum,1) = 0.0_rk; rhs(eqnnum) = 0.0_rk 
         amat(eqnnum,bgbl(2,bdelem,1,1,0,1)) = 1.0_rk; 
         rhs(eqnnum) = yl_soln(eqnnum+1) - this%z0
      case('natural')
      case default 
         write(stderr,*) ' Program aborted due to an incorrect option chosen. Only put "essential", "switch" and "natural" '
         stop
      end select
   end subroutine bc_b0r
   subroutine bc_b0z(this,eqnnum,bdelem)
      implicit none 
      class(yl_governers) :: this 
      integer(ik), intent(in) :: eqnnum, bdelem
      select case(this%b0_z)
      case('essential')
         amat(eqnnum,:) = 0.0_rk; colmat(eqnnum,1) = 0.0_rk; rhs(eqnnum) = 0.0_rk 
         amat(eqnnum,bgbl(2,bdelem,1,1,1,1)) = 1.0_rk;
         rhs(eqnnum) = yl_soln(eqnnum) - this%z0
      case('switch')
         amat(eqnnum,:) = 0.0_rk; colmat(eqnnum,1) = 0.0_rk; rhs(eqnnum) = 0.0_rk 
         amat(eqnnum,bgbl(2,bdelem,1,1,1,0)) = 1.0_rk;
         rhs(eqnnum) = yl_soln(eqnnum-1) - this%r0
      case('natural')
      case default 
         write(stderr,*) ' Program aborted due to an incorrect option chosen. Only put "essential", "switch" and "natural" '
         stop
      end select
   end subroutine bc_b0z
   subroutine bc_bfr(this,eqnnum,bdelem)
      implicit none 
      class(yl_governers) :: this
      integer(ik), intent(in) :: eqnnum, bdelem  
      select case(this%bf_r)
      case('essential')
         amat(eqnnum,:) = 0.0_rk; colmat(eqnnum,1) = 0.0_rk; rhs(eqnnum) = 0.0_rk 
         amat(eqnnum,bgbl(2,bdelem,3,3,0,0)) = 1.0_rk;
         rhs(eqnnum) = yl_soln(eqnnum) - this%rf
      case('switch')
         amat(eqnnum,:) = 0.0_rk; colmat(eqnnum,1) = 0.0_rk; rhs(eqnnum) = 0.0_rk 
         amat(eqnnum,bgbl(2,bdelem,3,3,0,1)) = 1.0_rk;
         rhs(eqnnum) = yl_soln(eqnnum+1) - this%zf
      case('natural')
      case default 
         write(stderr,*) ' Program aborted due to an incorrect option chosen. Only put "essential", "switch" and "natural" '
         stop
      end select
   end subroutine bc_bfr
   subroutine bc_bfz(this,eqnnum,bdelem)
      implicit none 
      class(yl_governers) :: this
      integer(ik), intent(in) :: eqnnum, bdelem  
      select case(this%bf_z)
      case('essential')
         amat(eqnnum,:) = 0.0_rk; colmat(eqnnum,1) = 0.0_rk; rhs(eqnnum) = 0.0_rk 
         amat(eqnnum,bgbl(2,bdelem,3,3,1,1)) = 1.0_rk;
         rhs(eqnnum) = yl_soln(eqnnum) - this%zf
      case('switch')
         amat(eqnnum,:) = 0.0_rk; colmat(eqnnum,1) = 0.0_rk; rhs(eqnnum) = 0.0_rk 
         amat(eqnnum,bgbl(2,bdelem,3,3,1,0)) = 1.0_rk;
         rhs(eqnnum) = yl_soln(eqnnum-1) - this%rf
      case('natural')
      case default 
         write(stderr,*) ' Program aborted due to an incorrect option chosen. Only put "essential", "switch" and "natural" '
         stop
      end select
   end subroutine bc_bfz
   subroutine l2_norm(iter,totnodes,errvec,toterr)
      implicit none 
      integer(ik), intent(in) :: iter,totnodes 
      real(rk), intent(in) :: errvec(:) 
      real(rk) :: toterr, error_r, error_z, error_vol 
      integer(ik) :: i 

      toterr = 0.0_rk 
      do i=1,size(errvec)
         toterr = toterr + errvec(i)**2 
      end do

      error_vol = errvec(size(errvec))
      error_r = 0.0_rk 
      error_z = 0.0_rk   
      do i=1,totnodes
         error_r = error_r + errvec(2*i-1)**2
         error_z = error_z + errvec(2*i+0)**2
      end do 
      write(stdout,*) "iter", iter, "Δr", error_r, "Δz", error_z, "Δp", error_vol 
   end subroutine l2_norm
   subroutine dessembler(solvec,mynodes,rvec,zvec)
      implicit none 
      real(rk), intent(in) :: solvec(:) 
      real(rk), intent(out) :: rvec(:), zvec(:)
      integer(ik), intent(in) :: mynodes
      integer(ik) :: n 

      do n=1,mynodes
         rvec(n) = solvec(2*n-1)
         zvec(n) = solvec(2*n+0) 
      end do 
      
   end subroutine dessembler
   subroutine obtain_initial_condition(solvec,np,initial_spline)
      implicit none  
      real(rk), intent(out) :: solvec(:) 
      integer(ik), intent(in) :: np 
      type(spline_objective) :: initial_spline  
      
      real(rk) :: delta_arc, acc_arc  
      integer(ik) :: i 
      
      delta_arc = maxval(initial_spline%sd)/real(np-1,rk)
      acc_arc = 0.0_rk 
      solvec(2*1-1) = initial_spline%xd(1)
      solvec(2*1-0) = initial_spline%yd(1)
      do i=2,np-1
         acc_arc = acc_arc + delta_arc  
         solvec(2*i-1) = spline_interp(initial_spline%sd,initial_spline%xd,initial_spline%sx2d,acc_arc)
         solvec(2*i-0) = spline_interp(initial_spline%sd,initial_spline%yd,initial_spline%sy2d,acc_arc)  
      end do 
      solvec(2*np-1) = initial_spline%xd(size(initial_spline%xd,dim=1))
      solvec(2*np-0) = initial_spline%yd(size(initial_spline%yd,dim=1))

   end subroutine obtain_initial_condition
   subroutine fill_in_rhs_amat_projection(n,this)
      implicit none 
      integer(ik), intent(in) :: n 
      type(yl_governers) :: this 
      real(rk) :: rloc(6), jloc(6,6) 
      integer(ik) :: k,i,j,gnode  

      real(rk) :: r_, z_, dr_dxi, dz_dxi 
      real(rk) :: s_xi, ds_xi_drj, ds_xi_dzj  
      real(rk) :: twoH, dtwoH_drj, dtwoH_dzj 
      real(rk) :: refK, drefK_drj, drefK_dzj
      real(rk) :: bonG, dbonG_drj, dbonG_dzj 

      integer(ik) :: dof1

      ! simple bookkeeping to keep in mind 
      !rhs=[r1,z1,r2,z2,r3,z3,r4,z4,r5,z5, ...]
      !ind=[1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9, 10, ...]
      !global node = 2*(n-1)+ local node 
      !dof_x = 2*global node-1
      !dof_y = 2*global node+0 
      ! equations
      !______________________________________________________________________________________! 
      !              Rᵢ =  ∫ [σ̃rrᵪ/sᵪ d/dχ(ϕⁱ) + σ̃ϕⁱsᵪ - (P₀ + Gz̃)r ϕⁱzᵪ] dχ                 ! 
      !                                   Rᵢ =  ∫ sᵪ ϕᵪⁱ dχ                                  !
      !______________________________________________________________________________________!
      ! twoH = σ̃rrᵪ/sᵪ d/dχ(ϕⁱ) + σ̃ϕⁱsᵪ 
      ! refK = - P₀r ϕⁱzᵪ 
      ! bonG = - Gz̃r ϕⁱzᵪ
      ! sᵪ = (rᵪ² + zᵪ²)^(1/2)

      ! initialize local residual and jacobian
      rloc = 0.0_rk 
      jloc = 0.0_rk 
      do k=1,3
         ! initialize variables to be interpolated
         r_ = 0.0_rk; z_ = 0.0_rk; dr_dxi = 0.0_rk; dz_dxi = 0.0_rk 
         ! interpolation 
         do i=1,3 
            gnode = 2*(n-1)+i
            r_ = r_ + yl_ph1(i,k)*yl_soln(2*gnode-1)
            z_ = z_ + yl_ph1(i,k)*yl_soln(2*gnode+0)
            dr_dxi = dr_dxi + yl_dph1(i,k)*yl_soln(2*gnode-1)
            dz_dxi = dz_dxi + yl_dph1(i,k)*yl_soln(2*gnode+0)
         end do 
         s_xi = sqrt(dr_dxi**2+dz_dxi**2) 
         do i=1,3 
            ! twoH = σ̃rrᵪ/sᵪ d/dχ(ϕⁱ) + σ̃ϕⁱsᵪ = σ̃ (rrᵪ/sᵪ d/dχ(ϕⁱ) + ϕⁱsᵪ) 
            twoH = this%sigma * ( r_*dr_dxi/s_xi*yl_dph1(i,k) + yl_ph1(i,k)*s_xi ) 
            ! refK = - P₀r ϕⁱzᵪ
            refK = - this%p0 * r_* yl_ph1(i,k) * dz_dxi
            ! bonG = - Gz̃r ϕⁱzᵪ
            bonG = - this%bond * z_ * r_ * yl_ph1(i,k) * dz_dxi 
            ! iunoi = local nopp = 2*i-1 
            rloc(2*i-1) = rloc(2*i-1) + yl_wt(k)*(twoH+refK+bonG)
            rloc(2*i+0) = rloc(2*i+0) + yl_wt(k)*(s_xi*yl_dph1(i,k))
            do j=1,3
               ds_xi_drj = dr_dxi*yl_dph1(j,k)/s_xi
               ds_xi_dzj = dz_dxi*yl_dph1(j,k)/s_xi 
               ! twoH = σ̃ (rrᵪ/sᵪ d/dχ(ϕⁱ) + ϕⁱsᵪ)  
               ! twoH = this%sigma * ( r_*dr_dxi/s_xi*yl_dph1(i,k) + yl_ph1(i,k)*s_xi ) 
               dtwoH_drj = this%sigma * ( yl_ph1(j,k)*dr_dxi/s_xi*yl_dph1(i,k) + r_*yl_dph1(j,k)/s_xi*yl_dph1(i,k) &
                                        + r_*dr_dxi/s_xi**2*yl_dph1(i,k)*(-ds_xi_drj) + yl_ph1(i,k)*ds_xi_drj )  
               dtwoH_dzj = this%sigma * ( r_*dr_dxi/s_xi**2*yl_dph1(i,k)*(-ds_xi_dzj) + yl_ph1(i,k)*ds_xi_dzj )   
               ! refK = - P₀r ϕⁱzᵪ 
               ! refK = - this%p0 * r_* yl_ph1(i,k) * dz_dxi
               drefK_drj = - this%p0 * yl_ph1(j,k)* yl_ph1(i,k) * dz_dxi
               drefK_dzj = - this%p0 * r_* yl_ph1(i,k) * yl_dph1(j,k)
               ! bonG = - Gz̃r ϕⁱzᵪ
               ! bonG = - this%bond * z_ * r_ * yl_ph1(i,k) * dz_dxi 
               dbonG_drj = - this%bond * z_ * yl_ph1(j,k) * yl_ph1(i,k) * dz_dxi 
               dbonG_dzj = - this%bond * yl_ph1(j,k) * r_ * yl_ph1(i,k) * dz_dxi & 
                           - this%bond * z_ * r_ * yl_ph1(i,k) * yl_dph1(j,k) 
               
               jloc(2*i-1,2*j-1) = jloc(2*i-1,2*j-1) + yl_wt(k)*(dtwoH_drj+drefK_drj+dbonG_drj)
               jloc(2*i-1,2*j+0) = jloc(2*i-1,2*j+0) + yl_wt(k)*(dtwoH_dzj+drefK_dzj+dbonG_dzj)
               jloc(2*i+0,2*j-1) = jloc(2*i+0,2*j-1) + yl_wt(k)*(ds_xi_drj*yl_dph1(i,k))
               jloc(2*i+0,2*j+0) = jloc(2*i+0,2*j+0) + yl_wt(k)*(ds_xi_dzj*yl_dph1(i,k))
            end do  
         end do  
      end do 

      ! global assembly 
      do i=1,3 
         gnode = 2*(n-1)+i
         dof1 = 2*gnode-1
         rhs(dof1+0) = rhs(dof1+0) + rloc(2*i-1) 
         rhs(dof1+1) = rhs(dof1+1) + rloc(2*i+0) 
         do j=1,3
            amat(dof1+0,bgbl(2,n,i,j,0,0)) = amat(dof1+0,bgbl(2,n,i,j,0,0)) + jloc(2*i-1,2*j-1)
            amat(dof1+0,bgbl(2,n,i,j,0,1)) = amat(dof1+0,bgbl(2,n,i,j,0,1)) + jloc(2*i-1,2*j+0)
            amat(dof1+1,bgbl(2,n,i,j,1,0)) = amat(dof1+1,bgbl(2,n,i,j,1,0)) + jloc(2*i+0,2*j-1)
            amat(dof1+1,bgbl(2,n,i,j,1,1)) = amat(dof1+1,bgbl(2,n,i,j,1,1)) + jloc(2*i+0,2*j+0)
         end do 
      end do 

   end subroutine fill_in_rhs_amat_projection
   subroutine fill_in_colmat_projection(n)
      implicit none 
      integer(ik), intent(in) :: n  
      real(rk) :: colloc(6) 
      integer(ik) :: k,i,gnode,dof1 
      real(rk) :: r_, dr_dxi, dz_dxi 
      real(rk) :: s_xi
 
      ! simple bookkeeping to keep in mind 
      !rhs=[r1,z1,r2,z2,r3,z3,r4,z4,r5,z5, ...]
      !ind=[1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9, 10, ...]
      !global node = 2*(n-1)+ local node 
      !dof_x = 2*global node-1
      !dof_y = 2*global node+0 
      ! equations
      !______________________________________________________________________________________! 
      !              Rᵢ =  ∫ [σ̃rrᵪ/sᵪ d/dχ(ϕⁱ) + σ̃ϕⁱsᵪ - (P₀ + Gz̃)r ϕⁱzᵪ] dχ                 ! 
      !                                   Rᵢ =  ∫ sᵪ ϕᵪⁱ dχ                                  !
      !______________________________________________________________________________________!
      ! sᵪ = (rᵪ² + zᵪ²)^(1/2)

      ! in this subroutine, we calculate dRᵢ/dP₀ = ∫ -r ϕⁱzᵪ dχ and dRᵢ/dP₀ = 0.0

      ! initialize local column
      colloc = 0.0_rk    
      do k=1,3
         ! initialize variables to be interpolated
         r_ = 0.0_rk; dr_dxi = 0.0_rk; dz_dxi = 0.0_rk 
         ! interpolation 
         do i=1,3 
            gnode = 2*(n-1)+i
            r_ = r_ + yl_ph1(i,k)*yl_soln(2*gnode-1)
            dr_dxi = dr_dxi + yl_dph1(i,k)*yl_soln(2*gnode-1)
            dz_dxi = dz_dxi + yl_dph1(i,k)*yl_soln(2*gnode+0)
         end do 
         s_xi = sqrt(dr_dxi**2+dz_dxi**2) 
         do i=1,3
            colloc(2*i-1) = colloc(2*i-1) + yl_wt(k)*(-r_*yl_ph1(i,k)*dz_dxi)
            colloc(2*i+0) = colloc(2*i+0) + 0.0_rk 
         end do 
      end do 

      do i=1,3
         dof1 = 2*(2*(n-1)+i)-1
         colmat(dof1+0,1) = colmat(dof1+0,1) + colloc(2*i-1) 
         colmat(dof1+1,1) = colmat(dof1+1,1) + colloc(2*i+0)  
      end do 

   end subroutine fill_in_colmat_projection 
   subroutine fill_in_rowmat_rhscon(n,this)
      implicit none 
      integer(ik), intent(in) :: n 
      type(yl_governers) :: this 
      real(rk) :: rowloc(6), vcon  
      integer(ik) :: k,i,j,gnode,dof2 
      real(rk) :: r_, dz_dxi 

      ! simple bookkeeping to keep in mind 
      !rhs=[r1,z1,r2,z2,r3,z3,r4,z4,r5,z5, ...]
      !ind=[1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9, 10, ...]
      !global node = 2*(n-1)+ local node 
      !dof_x = 2*global node-1
      !dof_y = 2*global node+0 
      ! equations
      !______________________________________________________________________________________!
      !                                                                                      !
      !                              Rᵥ = π∫ r²zᵪ dχ -  V₀                                   !
      !______________________________________________________________________________________! 

      ! in this subroutine, we calculate Rᵥ, dRᵥ/drj and dRᵥ/dzj 
      ! be careful; V₀ needs be subtracted only once, say when n = 1 

      ! initialize local row
      rowloc = 0.0_rk 
      vcon = 0.0_rk 
      do k=1,3
         ! initialize variables to be interpolated
         r_ = 0.0_rk; dz_dxi = 0.0_rk 
         ! interpolation 
         do i=1,3 
            gnode = 2*(n-1)+i
            r_ = r_ + yl_ph1(i,k)*yl_soln(2*gnode-1)
            dz_dxi = dz_dxi + yl_dph1(i,k)*yl_soln(2*gnode+0)
         end do 

         ! the last rhs slot 
         vcon = vcon + yl_wt(k)*pi*(r_**2*dz_dxi) 
         
         ! local row
         do j=1,3
            rowloc(2*j-1) = rowloc(2*j-1) + yl_wt(k)*pi*(2.0_rk*r_*dz_dxi*yl_ph1(j,k))
            rowloc(2*j+0) = rowloc(2*j+0) + yl_wt(k)*pi*(r_**2*yl_dph1(j,k)) 
         end do 
              
      end do 

      ! global assembly (volume constraint)
      rhs(size(rhs)) = rhs(size(rhs)) + vcon
      !if (n==2000) print*, rhs(size(rhs))
      if (n==1) rhs(size(rhs)) = rhs(size(rhs)) - this%v0
      ! global assembly (rowmat)
      do j=1,3
         dof2 = 2*(2*(n-1)+j)-1
         rowmat(1,dof2+0) = rowmat(1,dof2+0) + rowloc(2*j-1) 
         rowmat(1,dof2+1) = rowmat(1,dof2+1) + rowloc(2*j+0)  
      end do 

   end subroutine fill_in_rowmat_rhscon 
   subroutine yl_phi1d(gp3,temp_ph1,temp_dph1)
      implicit none
      real(rk), intent(in) :: gp3(:)
      real(rk), allocatable, intent(inout) :: temp_ph1(:,:), temp_dph1(:,:)
      integer(ik) :: j

      call alloc(temp_ph1,n=[3,3])
      call alloc(temp_dph1,n=[3,3])
      if (size(gp3)/=3) then 
         write(stderr,*) "size of gauss points must be 3 (check yl_phi1d in young laplacer)"
         stop  
      end if 

      do j=1,3
         temp_ph1(1,j)  = 1.0_rk - 3.0_rk*gp3(j) + 2.0_rk*gp3(j)*gp3(j)
         temp_dph1(1,j) = -3.0_rk + 4.0_rk*gp3(j)
         temp_ph1(2,j)  = 4.0_rk*(gp3(j) - gp3(j)*gp3(j))
         temp_dph1(2,j) = 4.0_rk - 8.0_rk*gp3(j)
         temp_ph1(3,j)  = - gp3(j) + 2.0_rk*gp3(j)*gp3(j)
         temp_dph1(3,j) = -1.0_rk + 4.0_rk*gp3(j) 
      end do
   end subroutine yl_phi1d
   integer(ik) function bgbl(m,n,i,j,k,l)
      implicit none 
      integer(ik), intent(in) :: m,n,i,j,k,l 
      integer(ik) :: global_node1, global_node2
      integer(ik) :: nopp1, nopp2
      integer(ik) :: dof1, dof2     
      ! i and j only go over 1,2,3 
      global_node1 = 2*(n-1)+i
      global_node2 = 2*(n-1)+j
      ! it is assumed that only 2 dof are present at each node   
      nopp1 = 2*global_node1-1 
      nopp2 = 2*global_node2-1 
      dof1 = nopp1 + k 
      dof2 = nopp2 + l  
      if (m==1) then
          bgbl = dof1  
      elseif (m==2) then
          bgbl = dof2 - dof1 + (myband+1)/2 
      else 
          write(stderr, *) 'size(bgbl,dim=1) must be 2'
          stop
      end if 
   end function bgbl
!***********************************************************************************************************************
! The Arrow Solver  
!***********************************************************************************************************************
!-----------------------------------------------------------------------| 
!       WRITTEN BY:             PAUL THOMAS     (617) 253-6547          | 
!                               MIT 66-256                              | 
!                               77 MASSACHUSETTS AVENUE                 | 
!                               CAMBRIDGE, MA  02139                    | 
!                                                                       | 
!       THIS SUBROUTINE SOLVES THE MATRIX EQUATION:     A * X = B       | 
!                                                                       | 
!       WHERE A IS AN ARROW MATRIX AND B,X ARE VECTORS.  AN ARROW       | 
!       MATRIX HAS NONZERO ELEMENTS ONLY IN A NARROW BAND CENTERED      | 
!       ON THE MAIN DIAGONAL AND IN THE LAST FEW COLUMNS AND ROWS.      | 
!                                                                       | 
!       THE MATRIX A IS REDUCED USING GAUSSIAN ELIMINATION WITH         | 
!       THRESHOLD ROW PIVOTING.  THE COLUMN IN THE BAND MATRIX IS       | 
!       SEARCHED FOR THE MAXIMUM ELEMENT.  IF THE DIAGONAL ELEMENT      | 
!       IS LESS THAN A FRACTION(TAU) OF THE MAXIMUM ELEMENT, THEN       | 
!       THE ROWS ARE EXCHANGED.                                         | 
!                                                                       | 
!       TAU = 1.0       PARTIAL PIVOTING                                | 
!       TAU = 0.2-0.3   RECOMMENDED RANGE                               | 
!       TAU = 0.0       NO PIVOTING                                     | 
!       TAU = -1.0      SOLVES L U X = B.  INPUT IS MATRIX PREVIOUSLY   | 
!                       FACTORED WITH TAU = 0.0                         | 
!                                                                       | 
!       THE PROGRAM WILL RETURN THE CORRECT  L U DECOMPOSITION IF IT    | 
!       HAS NOT PIVOTED, WHICH IS INDICATED WHEN NPIV=0.  THIS CAN      | 
!       BE FORCED BY SETTING TAU=0.0, SUPPRESSING PIVOTING.  THE        | 
!       MATRIX EQUATION IS SOLVED FOR MULTIPLE B VECTORS.  THE          | 
!       MAGNITUDE AND SIGN OF THE DETERMINANT ARE ALSO CALCULATED.      | 
!       NOTE:  EXTRA STORAGE IS REQUIRED FOR THE BAND MATRIX.           | 
!-----------------------------------------------------------------------| 
! 
!       CODE STRUCTURE MODERNIZED (e.g. END DO instead of CONTINUE) 
!       BY HAK KOON YEOH, 03 APRIL 2006, PURDUE UNIVERSITY 
! 

subroutine arrowsolver(nod, iband, iarrow, tau) 
   implicit none 
   integer(ik), intent(in) :: nod, iband, iarrow 
   real(rk), intent(in) :: tau    
   
   integer(ik) :: isign, npiv, mid, mex, nex 
   integer(ik) :: kks, lls, ms, is1, js1, ks1, js2 
   integer(ik) :: is, js
   
   real(rk) :: detlog, big, trans, diag, piv  
  
   ! nod: dimension of the banded matrix A and of the columns and rows of the arrow 
   ! iband: band width of matrix A, ``must be odd'' 
   ! iarrow: width of the columns and rows in the arrow (number of constraints)
   ! tau: minimum ratio of diagonal to maximum element
   !      for no pivoting and LU, set tau=0.0 
   !      for full pivoting, set tau=1.0 
   !      for solving LUX=B, set tau=-1.0 
   ! detlog: log 10 of the determinant 
   ! isign: sign of the determinant 
   ! npiv: number of times rows have been pivoted 

   ! rhs(nod+iarrow): holds B vector. The solution vector X is returned in this vector 
   ! head(iarrow,iarrow): holds the elements in the square section of the arrow matrix 
   ! rowmat(iarrow,nod): holds the elements of the rows of the arrow matrix 
   ! colmat(nod,iarrow): holds the elements of the columns of the arrow matrix 
   ! amat(nod,(3*iband-1)/2): banded matrix A. Note that the dimensions are 50% greater than the bandwidth 
 
   mid = (iband+1)/2 
   mex = 0
   nex = 0
   detlog = 0.0_rk 
   isign = 1  
   npiv = 0 

   !------------------------------------------------------------------------------------------------
   ! amat(i,mid): contains the diagonal elements of matrix A
   ! mex: keeps track of the maximum row exchange in the matrix A 
   ! nex: keeps track of the last non-zero element in the row in matrix A 
   !
   ! clean out matrix A outside the band width
   ! note iband+mid-1 = iband+(iband+1)/2-1 = [2*iband-2+iband+1]/2 = (3*iband-1)/2
   !------------------------------------------------------------------------------------------------
   do is=1,nod
      do js=iband+1,iband+mid-1
         amat(is,js) = 0.0_rk  
      end do  
   end do 

   ! gaussian elimination of matrix A. 
   ! check the column for the maximum element. 
   ! if the diagonal is less than a fraction, tau, of the maximum element, then switch rows. 
   ! if tau=0.0, then no pivoting will take place and the LU decomposition will be returned.
   ! if tau=1.0, then full row pivoting will be used. 
   ! if tau=-1.0, then the LU matrix will be front-substituted, and then back-substituted. 
  
   if (tau<0.0) then 
      ! if tau<0.0, the matrix is not decomposed, and the previous LU matrix is used. 
      ! the lower matrix L is forward substituted. 

      do is=1, nod 
         is1 = min(is+mid-1,nod)
         piv = rhs(is)
         do js=is+1,is1
            rhs(js) = rhs(js) - piv*amat(js,is+mid-js) 
         end do 
         do js=1,iarrow 
            rhs(nod+js) = rhs(nod+js) - piv*rowmat(js,is)
         end do 
      end do 

      do is= 1, iarrow 
         piv = rhs(nod+is)
         do js= is+1,iarrow
            rhs(nod+js) = rhs(nod+js) - piv*head(js,is) 
         end do 
      end do 

   else 

      ! if tau>0.0 
      do is=1, nod-1
         
         lls = is 
         big = abs(amat(is,mid))

         do js=is+1,min(lls+mid-1,nod)
            if (abs(amat(js, mid+is-js))<big) cycle 
            big = abs(amat(js, mid+is-js))
            lls = js 
         end do 

         if (abs(amat(is,mid))<tau*big) then
            ! exchange row I with tow LL 
            ! the vector B and array COL are exchanged 
            ! the determinant of the matrix changes sign. 
            
            mex = max(mex, lls - is)
            nex = max(nex, lls - is)
            ks1 = min(iband+nex,nod-is+mid)

            do js=mid,ks1
               trans = amat(is,js)
               amat(is,js) = amat(lls, js+is-lls)
               amat(lls,js+is-lls) = trans  
            end do 

            do js=1,iarrow
               trans = colmat(is,js)
               colmat(is,js) = colmat(lls,js)
               colmat(lls,js) = trans  
            end do 

            trans = rhs(is)
            rhs(is) = rhs(lls)
            rhs(lls) = trans 

            isign = -isign 
            npiv = npiv + 1 
         end if

         ! the lower band matrix is eliminated and replaced with the pivots for the LU decomposition.
         ! the B vector and the array COL is updated. 

         diag = amat(is,mid)
         nex = max(lls-is,nex-1)
         js1 = max(1,is+mid-nod)
         js2 = max(1-nex,is+mid-nod)

         do js=mid-1,js1,-1

            is1 = mid + is - js 
            piv = amat(is1,js)/diag 
            amat(is1,js) = piv 
            
            rhs(is1) = rhs(is1) - piv*rhs(is) 

            do ms=1,iarrow
               colmat(is1,ms) = colmat(is1,ms) - piv*colmat(is,ms)
            end do 

            do kks=js+1,js+mid-js2 
               amat(is1,kks) = amat(is1,kks) - piv*amat(is,mid+kks-js)
            end do 

         end do 
 
      end do
      
      ! the array row is eliminated except for the last column and replaced with the pivot for the LU decomposition.
      ! the B vector and the array head are updated. 

      do is=1,nod-1
         
         diag = amat(is,mid)
         js1 = min(is+mid-1+mex,nod)

         do js=1, iarrow 
            
            piv = rowmat(js,is)/diag 
            rowmat(js,is)=piv 
            
            rhs(nod+js) = rhs(nod+js) - piv*rhs(is)

            do kks=is+1,js1 
               rowmat(js,kks) = rowmat(js,kks) - piv*amat(is, mid+kks-is)
            end do 

            do lls=1,iarrow 
               head(js,lls) = head(js,lls) - piv*colmat(is,lls)
            end do 

         end do 

      end do 

      ! the final element of the banded matrix A is checked to see if it is large enough.
      ! if it is too small, it is pivoted with an element from the last column of the array row. 

      big = abs(amat(nod,mid))
      do js=1,iarrow
         if (abs(rowmat(js,nod))<big) cycle
         
         big = abs(rowmat(js,nod))
         lls = js 
      end do 

      if (abs(amat(nod,mid))<tau*big.and.lls<=iarrow) then
         
         trans = rhs(nod)
         rhs(nod) = rhs(nod+lls)
         rhs(nod+lls) = trans 
         
         do js=1,iarrow
            trans = colmat(nod,js)
            colmat(nod,js) = head(lls,js)
            head(lls,js) = trans  
         end do 

         trans = amat(nod,mid)
         amat(nod,mid) = rowmat(lls,nod)
         rowmat(lls,nod) = trans 

         isign = -isign 
         npiv = npiv + 1 

      end if 

      ! the last column of the array row is eliminated. 
      ! the B vector and the array head are updated. 

      diag = amat(nod,mid)

      do is=1,iarrow 

         piv = rowmat(is,nod)/diag 
         rowmat(is,nod) = piv 

         rhs(nod+is) = rhs(nod+is) - piv*rhs(nod)

         do js=1,iarrow 
            head(is,js) = head(is,js) - piv*colmat(nod,js)
         end do 

      end do 

      ! the array head is eliminated using threshold row pivoting.
      ! if the diagonal element is less than A fraction, tau, of the maximum column element,
      ! then the rows are exchanged. 
      ! as the column is eliminated, the B vector and the array head are updated. 

      do is=1,iarrow-1 

         big = abs(head(is,is))

         do js=is+1,iarrow 
            if (abs(head(js,is))<big) cycle 
            big = abs(head(js,is))
            lls = js  
         end do 

         if (abs(head(is,is))<tau*big) then 

            trans = rhs(nod+is)
            rhs(nod+is) = rhs(nod+lls)
            rhs(nod+lls) = trans 

            do js=1,iarrow
               trans=head(is,js)
               head(is,js) = head(lls,js)
               head(lls,js) = trans  
            end do 

            isign = -isign 
            npiv = npiv + 1 

         end if 

         diag = head(is,is)

         do js=is+1,iarrow

            piv = head(js,is)/diag 
            head(js,is) = piv 
            
            rhs(nod+js) = rhs(nod+js) - piv*rhs(nod+is) 

            do kks=is+1,iarrow
               head(js,kks) = head(js,kks) - piv*head(is,kks) 
            end do 

         end do 

      end do 

   end if 

   ! the upper matrix U containing the arrow is back substituted for the solution vectors X.
   ! The magnitude and sign of the matrix are calculated. 

   do is=iarrow,1,-1 

      diag = head(is,is)
      detlog = detlog + log10(abs(diag))

      if (diag<0.0) isign = -isign 

      piv = rhs(nod+is)/diag 
      rhs(nod+is) = piv 

      do js=is-1,1,-1 
         rhs(nod+js) = rhs(nod+js) - piv*head(js,is)
      end do 

      do kks=nod,1,-1
         rhs(kks) = rhs(kks) - piv*colmat(kks,is) 
      end do 
   end do 

   ! the upper matrix U containing the banded matrix A is back-substituted for the solution vectors X 
   ! the magnitude and sign of the determinant are calculated.

   do is=nod,1,-1
      
      diag = amat(is,mid)
      detlog = detlog + log10(abs(diag))

      if (diag<0.0) isign = -isign 

      js1 = max(1,is-mid-1-mex)

      piv = rhs(is)/diag 
      rhs(is) = piv 

      do js=is-1,js1,-1
         rhs(js) = rhs(js) - piv*amat(js, mid+is-js) 
      end do 

   end do 

   return
end subroutine arrowsolver

pure function spline_interp(xa,ya,y2a,x) result(res)
            real(rk), intent(in) :: xa(:), ya(:), y2a(:), x
            real(rk) :: res
        ! Given the arrays xa(1:n) and ya(1:n) of length n, which tabulate a function
        ! (with the xa_i’s in order), and given the array y2a(1:n), which is the output
        ! from make_2nd_order_deriv below, and given a value of x,
        ! this routine returns a cubic-spline interpolated value y.
        integer(ik) :: khi, klo, n, k
        real(rk) :: a, b, h

        n = size(xa)
        !klo = max(min(locate(xa,x),n-1),1)
        !khi = klo + 1
        klo = 1
        khi = n
        do
            if (khi-klo.gt.1) then
                k = (khi+klo)/2
                if (xa(k) > x) then
                    khi = k
                else
                    klo = k
                endif
            else
                exit
            end if
        end do
        h = xa(khi) - xa(klo)
        a = (xa(khi)-x)/h
        b = (x-xa(klo))/h
        res = a*ya(klo) + b*ya(khi) + ((a**3-a)*y2a(klo)+(b**3-b)*y2a(khi))*(h**2)/6.0_rk
end function spline_interp

subroutine make_s_from_xy(tab_x,tab_y,indep_s)
            real(rk), intent(in) :: tab_x(:), tab_y(:)
            real(rk), allocatable, intent(out) :: indep_s(:)
        ! this subroutines compute arc length from (x_{i},y_{i}), i=1, ..., N
        ! x and y can then be viewed as dependent variables
        ! with respect to the independent variable arc length s
        ! (dependent=x,independent=y) -> (dependent=s,independent=x)
        ! and (dependent=s,independent=y)
        real(rk) :: dsdet, dxdet, dydet, s_i
        real(rk) :: ds1det, dx1det, dy1det, s_i_1
        integer(ik) :: n, m, i, j, k, ele
        real(rk), parameter :: wt_(3) = [(0.555555555555556_rk)*0.5_rk, &
                                        (0.888888888888889_rk)*0.5_rk,&
                                        (0.555555555555556_rk)*0.5_rk]
        real(rk), parameter :: ai_(3) = [(1.0_rk-0.774596669241483_rk)*0.5_rk, &
                                        0.5_rk,(1.0_rk+0.774596669241483_rk)*0.5_rk]

        n = size(tab_x)
        if (n/=size(tab_y)) then
            write(stderr, *) 'the size of tab_x must be equal to that of tab_y'
            write(stderr, *) 'check subroutine arc_length_from_xy'
            stop
        end if
        if (mod(n,2)==0) then
            write(stderr, *) 'the size of tab_x must be odd'
            write(stderr, *) 'check subroutine arc_length_from_xy'
            stop
        end if

        ! allocate arc length vector
        call alloc(indep_s,n)

        ! take finite element integration approach
        ! m is the total number of effective elements
        m = (n-1)/2

        ! arc length is always measured from 0
        indep_s(1) = 0.0_rk

        do ele=1,m
            s_i = 0.0_rk; s_i_1 = 0.0_rk;
            do k=1,3
                dxdet = 0.0_rk; dydet = 0.0_rk;
                dx1det = 0.0_rk; dy1det = 0.0_rk;
                do i=1,3
                    j = 2*(ele-1)+i
                    dxdet  = dxdet  + dquadb1(i,ai_(k)       )*tab_x(j)
                    dydet  = dydet  + dquadb1(i,ai_(k)       )*tab_y(j)
                    dx1det = dx1det + dquadb1(i,ai_(k)*0.5_rk)*tab_x(j)
                    dy1det = dy1det + dquadb1(i,ai_(k)*0.5_rk)*tab_y(j)
                End do
                dsdet  = sqrt(dxdet**2+dydet**2)
                ds1det = sqrt(dx1det**2+dy1det**2)
                s_i   = s_i   + wt_(k)*dsdet
                s_i_1 = s_i_1 + wt_(k)*0.5_rk*ds1det
            End do
            indep_s(2*ele  ) = indep_s(2*ele-1) + s_i_1
            indep_s(2*ele+1) = indep_s(2*ele-1) + s_i
        end do
end subroutine make_s_from_xy

subroutine make_2nd_order_deriv(tab_x,tab_y,yp1,ypn,y2)
            real(rk), intent(in) :: tab_x(:), tab_y(:), yp1, ypn
            real(rk), allocatable, intent(out) :: y2(:)
        ! Given arrays tab_x(1:n) and tab_y(1:n) containing a tabulated function,
        ! i.e., yi = f (xi ), with x1 <x2 < ... <xN, and given values yp1 and ypn for
        ! the first derivative of the interpolating function at points 1 and n,
        ! respectively, this routine returns an array y2(1:n) of length n which contains
        ! the second derivatives of the interpolating function at the tabulated points xi.
        ! If yp1 and/or ypn are equal to 1.0E+30 or larger, the routine is signaled
        ! to set the corresponding boundary condition for a natural spline,
        ! with zero second derivative on that boundary.
        integer(ik) :: n
        real(rk), allocatable :: a(:), b(:), c(:), r(:)

        ! the number of points
        n = size(tab_x)
        ! allocate aux arrays
        call alloc(a,n)
        call alloc(b,n)
        call alloc(c,n)
        call alloc(r,n)
        ! allocate output array
        call alloc(y2,n)

        c(1:n-1) = tab_x(2:n)-tab_x(1:n-1)
        r(1:n-1) = 6.0_rk*((tab_y(2:n)-tab_y(1:n-1))/c(1:n-1))
        r(2:n-1) = r(2:n-1)-r(1:n-2)
        a(2:n-1) = c(1:n-2)
        b(2:n-1) = 2.0_rk*(c(2:n-1)+a(2:n-1))
        b(1) = 1.0_rk
        b(n) = 1.0_rk

        if (yp1 > 0.99E30_rk) then ! the lower boundary condition is set to be natural
            r(1)=0.0_rk
            c(1)=0.0_rk
        else ! the lower boundary condition is set to have a specified first derivative
            r(1)=(3.0_rk/(tab_x(2)-tab_x(1)))*((tab_y(2)-tab_y(1))/(tab_x(2)-tab_x(1))-yp1)
            c(1)=0.5_rk
        end if
        if (ypn > 0.99E30_rk) then ! the upper boundary condition is set to be natural
            r(n)=0.0_rk
            a(n)=0.0_rk
        else ! the upper boundary condition is set to have a specified first derivative
            r(n)=(-3.0_rk/(tab_x(n)-tab_x(n-1)))*((tab_y(n)-tab_y(n-1))/(tab_x(n)-tab_x(n-1))-ypn)
            a(n)=0.5_rk
        end if
        ! solve with tridiagonal algorithm
        call tridag_par(a(2:n),b(1:n),c(1:n-1),r(1:n),y2(1:n))
end subroutine make_2nd_order_deriv

subroutine tridag_ser(a,b,c,r,u)
            real(rk), intent(in) :: a(:), b(:), c(:), r(:)
            real(rk), intent(out) :: u(:)
        real(rk), allocatable :: gam(:)
        integer(ik) :: n,j
        real(rk) :: bet

        n = size(b)
        call alloc(gam,n)
        bet=b(1)
        u(1)=r(1)/bet
        do j=2,n
            gam(j) = c(j-1)/bet
            bet = b(j) - a(j-1)*gam(j)
            u(j)=(r(j)-a(j-1)*u(j-1))/bet
        end do
        do j=n-1,1,-1
            u(j)=u(j)-gam(j+1)*u(j+1)
        end do
end subroutine tridag_ser

recursive subroutine tridag_par(a,b,c,r,u)
            real(rk), intent(in) :: a(:), b(:), c(:), r(:)
            real(rk), intent(out) :: u(:)
        integer(ik), parameter :: npar_tridag=4
        integer(ik) :: n, n2, nm, nx
        real(rk), allocatable :: y(:),q(:),piva(:),x(:),z(:),pivc(:)

        n = size(b)
        if (n<npar_tridag) then
            call tridag_ser(a,b,c,r,u)
        else
            call alloc(y,n/2)
            call alloc(q,n/2)
            call alloc(piva,n/2)
            call alloc(x,n/2-1)
            call alloc(z,n/2-1)
            call alloc(pivc,size(a)/2)

            n2 = size(y)
            nm = size(pivc)
            nx = size(x)

            piva = a(1:n-1:2)/b(1:n-1:2)
            pivc = c(2:n-1:2)/b(3:n:2)
            y(1:nm) = b(2:n-1:2)-piva(1:nm)*c(1:n-2:2)-pivc*a(2:n-1:2)
            q(1:nm) = r(2:n-1:2)-piva(1:nm)*r(1:n-2:2)-pivc*r(3:n:2)

            if (nm < n2) then
                y(n2) = b(n)-piva(n2)*c(n-1)
                q(n2) = r(n)-piva(n2)*r(n-1)
            end if

            x = -piva(2:n2)*a(2:n-2:2)
            z = -pivc(1:nx)*c(3:n-1:2)

            call tridag_par(x,y,z,q,u(2:n:2))

            u(1) = (r(1)-c(1)*u(2))/b(1)
            u(3:n-1:2) = (r(3:n-1:2)-a(2:n-2:2)*u(2:n-2:2) -c(3:n-1:2)*u(4:n:2))/b(3:n-1:2)
            if (nm == n2) u(n)=(r(n)-a(n-1)*u(n-1))/b(n)
        end if
end subroutine tridag_par

function dquadb1(local_nodes, xi_)
            integer(ik), intent(in) :: local_nodes
            real(rk) :: xi_
            real(rk) :: dquadb1
        if (local_nodes==1) then
            dquadb1 = - 3.0_rk + 4.0_rk*xi_
        else if (local_nodes==2) then
            dquadb1 = 4.0_rk - 8.0_rk*xi_
        else if (local_nodes==3) then
            dquadb1 = - 1.0_rk + 4.0_rk*xi_
        end if
end function dquadb1

subroutine meshwriter_1d(filename,t,xd,yd)
            character(len=*) :: filename
            integer(ik), intent(in) :: t
            real(rk), intent(in) :: xd(:), yd(:)
        integer(ik) :: openunit
        integer(ik) :: i, n
        character(len=:), allocatable :: fmt
        logical :: exists

        fmt = '(2es24.15)'

        n = size(xd)
        if (n/=size(yd)) then
            write(stderr, *) 'the size of xd must be equal to that of yd'
            write(stderr, *) 'check subroutine meshwriter_1d'
            error stop
        end if

        inquire(file=adjustl(filename), exist=exists)
        if (t.eq.1) then
            Open (newunit=openunit,file=adjustl(filename),Status='Replace')
        else
            if (exists) then 
                Open (newunit=openunit,file=adjustl(filename),Status='old',position='append',action='write')
            else 
                Open (newunit=openunit,file=adjustl(filename),Status='Replace')
            end if 
        endif

        Write(openunit,'(A)') 'Variables = "x","y"'
        Write(openunit,'(A,i5,A,A,i6,A,i5)') &
                        'Zone T = " ',t, '", F = Point,',' I=',n,' DT = (double,double)'
        
        do i = 1, n
            Write (openunit,fmt) xd(i), yd(i)
        end do

        close(openunit)
end subroutine meshwriter_1d

End module mod_yl 