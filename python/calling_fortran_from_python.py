get_ipython().system('pip install fortran-magic')

import warnings
warnings.filterwarnings('ignore')

get_ipython().magic('load_ext fortranmagic')

get_ipython().run_cell_magic('fortran', '', '\nsubroutine fib3(n, a)\n    integer, intent(in) :: n\n    real, intent(out) :: a\n\n    integer :: i\n    real :: b, tmp\n\n    a = 0\n    b = 1\n    do i = 1, n\n        tmp = b\n        b = a\n        a = a + tmp\n    end do\nend subroutine')

fib3(100)



