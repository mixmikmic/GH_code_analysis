import nose

nose.__version__

get_ipython().system('nosetests -vv --collect-only')

get_ipython().system('nosetests -vv --collect-only')

get_ipython().system('cd .. && nosetests -vv --collect-only')

get_ipython().system('cd .. && nosetests -vv --collect-only tests.tests01_geom')

get_ipython().system('cd .. && nosetests -vv --collect-only --with-timer tests.tests01_geom')

get_ipython().system('cd .. && python -W ignore -m nose -v --logging-filter=SAWarning --with-timer tests.tests01_geom')

4000 / 60

"""[success] 21.88% tests.tests01_geom.tests03_core.Test15_DetectLensTor.test12_plot_Etend_AlongLOS: 459.2323s
[success] 14.52% tests.tests01_geom.tests03_core.Test14_DetectApertLin.test12_plot_Etend_AlongLOS: 304.8276s
[success] 14.24% tests.tests01_geom.tests03_core.Test20_GDetectLensLin.test10_plot_Etend_AlongLOS: 298.8144s
[success] 12.96% tests.tests01_geom.tests03_core.Test19_GDetectLensTor.test10_plot_Etend_AlongLOS: 271.9052s
[success] 11.09% tests.tests01_geom.tests03_core.Test16_DetectLensLin.test12_plot_Etend_AlongLOS: 232.6831s
[success] 6.63% tests.tests01_geom.tests03_core.Test14_DetectApertLin.test05_debug_Etendue_BenchmarkRatioMode: 139.1220s
[success] 6.43% tests.tests01_geom.tests03_core.Test15_DetectLensTor.test05_debug_Etendue_BenchmarkRatioMode: 134.8452s
[success] 5.53% tests.tests01_geom.tests03_core.Test16_DetectLensLin.test05_debug_Etendue_BenchmarkRatioMode: 116.1267s
[success] 2.04% tests.tests01_geom.tests03_core.Test20_GDetectLensLin.test07_set_Res: 42.7960s
[success] 1.95% tests.tests01_geom.tests03_core.Test19_GDetectLensTor.test07_set_Res: 41.0061s
[success] 0.67% tests.tests01_geom.tests03_core.Test16_DetectLensLin.test08_set_Res: 14.0677s
[success] 0.52% tests.tests01_geom.tests03_core.Test15_DetectLensTor.test08_set_Res: 10.9890s
[success] 0.43% tests.tests01_geom.tests03_core.Test20_GDetectLensLin.test09_plot_SAngNb: 9.1035s
[success] 0.28% tests.tests01_geom.tests03_core.Test19_GDetectLensTor.test09_plot_SAngNb: 5.8537s
[success] 0.07% tests.tests01_geom.tests03_core.Test14_DetectApertLin.test08_set_Res: 1.4101s
[success] 0.04% tests.tests01_geom.tests03_core.Test05_LOSTor.test01_plot: 0.8269s
[success] 0.03% tests.tests01_geom.tests03_core.Test16_DetectLensLin.test10_plot_SAngNb: 0.6352s
[success] 0.03% tests.tests01_geom.tests03_core.Test04_StructLin.test02_plot: 0.5558s
[success] 0.03% tests.tests01_geom.tests03_core.Test03_StructTor.test02_plot: 0.5541s
[success] 0.03% tests.tests01_geom.tests03_core.Test14_DetectApertLin.test10_plot_SAngNb: 0.5348s
[success] 0.03% tests.tests01_geom.tests03_core.Test15_DetectLensTor.test10_plot_SAngNb: 0.5345s
[success] 0.03% tests.tests01_geom.tests03_core.Test12_ApertLin.test01_plot: 0.5331s
[error] 0.02% tests.tests01_geom.tests03_core.Test15_DetectLensTor.test11_debug_plot_SAng_OnPlanePerp: 0.4951s
[success] 0.02% tests.tests01_geom.tests03_core.Test15_DetectLensTor.test15_saveload: 0.4241s
[error] 0.02% <nose.suite.ContextSuite context=Test18_GDetectApertLin>:setup: 0.3997s
[error] 0.02% <nose.suite.ContextSuite context=Test17_GDetectApertTor>:setup: 0.3937s
[success] 0.02% tests.tests01_geom.tests03_core.Test14_DetectApertLin.test15_saveload: 0.3903s
[success] 0.02% tests.tests01_geom.tests03_core.Test16_DetectLensLin.test15_saveload: 0.3807s
[success] 0.02% tests.tests01_geom.tests03_core.Test15_DetectLensTor.test09_plot: 0.3734s
[success] 0.02% tests.tests01_geom.tests03_core.Test07_GLOSTor.test02_plot: 0.3597s
[success] 0.02% tests.tests01_geom.tests03_core.Test06_LOSLin.test01_plot: 0.3510s
[error] 0.02% tests.tests01_geom.tests03_core.Test16_DetectLensLin.test11_debug_plot_SAng_OnPlanePerp: 0.3282s
[success] 0.01% tests.tests01_geom.tests03_core.Test16_DetectLensLin.test09_plot: 0.2946s
[success] 0.01% tests.tests01_geom.tests03_core.Test09_LensTor.test02_plot: 0.2909s
[success] 0.01% tests.tests01_geom.tests03_core.Test08_GLOSLin.test02_plot: 0.2864s
[success] 0.01% tests.tests01_geom.tests03_core.Test20_GDetectLensLin.test08_plot: 0.2857s
[success] 0.01% tests.tests01_geom.tests03_core.Test16_DetectLensLin.test13_plot_Sinogram: 0.2830s
[success] 0.01% tests.tests01_geom.tests03_core.Test14_DetectApertLin.test09_plot: 0.2802s
[error] 0.01% tests.tests01_geom.tests03_core.Test14_DetectApertLin.test11_debug_plot_SAng_OnPlanePerp: 0.2716s
[success] 0.01% tests.tests01_geom.tests03_core.Test01_VesTor.test04_plot: 0.2700s
[success] 0.01% tests.tests01_geom.tests03_core.Test01_VesTor.test05_plot_Sinogram: 0.2689s
[success] 0.01% tests.tests01_geom.tests03_core.Test02_VesLin.test03_plot: 0.2666s
[success] 0.01% tests.tests01_geom.tests03_core.Test02_VesLin.test05_plot_Sinogram: 0.2575s
[success] 0.01% tests.tests01_geom.tests03_core.Test19_GDetectLensTor.test08_plot: 0.2505s
[success] 0.01% tests.tests01_geom.tests03_core.Test10_LensLin.test02_plot: 0.2451s
[success] 0.01% tests.tests01_geom.tests03_core.Test20_GDetectLensLin.test11_plot_Sinogram: 0.2420s
[success] 0.01% tests.tests01_geom.tests03_core.Test14_DetectApertLin.test14_plot_Res: 0.2293s
[success] 0.01% tests.tests01_geom.tests03_core.Test15_DetectLensTor.test14_plot_Res: 0.2292s
[success] 0.01% tests.tests01_geom.tests03_core.Test20_GDetectLensLin.test14_plot_Res: 0.2290s
[success] 0.01% tests.tests01_geom.tests03_core.Test16_DetectLensLin.test14_plot_Res: 0.2252s
[success] 0.01% tests.tests01_geom.tests03_core.Test19_GDetectLensTor.test14_plot_Res: 0.2232s
[success] 0.01% tests.tests01_geom.tests03_core.Test05_LOSTor.test02_plot_Sinogram: 0.2207s
[success] 0.01% tests.tests01_geom.tests03_core.Test11_ApertTor.test01_plot: 0.2203s
[success] 0.01% tests.tests01_geom.tests03_core.Test19_GDetectLensTor.test11_plot_Sinogram: 0.1888s
[success] 0.01% tests.tests01_geom.tests03_core.Test07_GLOSTor.test03_plot_Sinogram: 0.1852s
[success] 0.01% tests.tests01_geom.tests03_core.Test08_GLOSLin.test03_plot_Sinogram: 0.1838s
[success] 0.01% tests.tests01_geom.tests03_core.Test19_GDetectLensTor.test15_saveload: 0.1823s
[success] 0.01% tests.tests01_geom.tests03_core.Test06_LOSLin.test02_plot_Sinogram: 0.1802s
[success] 0.01% tests.tests01_geom.tests03_core.Test15_DetectLensTor.test13_plot_Sinogram: 0.1762s
[success] 0.01% tests.tests01_geom.tests03_core.Test14_DetectApertLin.test13_plot_Sinogram: 0.1728s
[success] 0.01% tests.tests01_geom.tests03_core.Test10_LensLin.test01_plot_alone: 0.1641s
[success] 0.01% tests.tests01_geom.tests03_core.Test09_LensTor.test01_plot_alone: 0.1568s
[success] 0.00% tests.tests01_geom.tests03_core.Test20_GDetectLensLin.test15_saveload: 0.0883s
[success] 0.00% tests.tests01_geom.tests03_core.Test04_StructLin.test03_saveload: 0.0510s
[success] 0.00% tests.tests01_geom.tests03_core.Test19_GDetectLensTor.test12_plot_Etendues: 0.0497s
[success] 0.00% tests.tests01_geom.tests03_core.Test20_GDetectLensLin.test12_plot_Etendues: 0.0490s
[success] 0.00% tests.tests01_geom.tests03_core.Test07_GLOSTor.test04_saveload: 0.0457s
[success] 0.00% tests.tests01_geom.tests03_core.Test08_GLOSLin.test04_saveload: 0.0445s
[error] 0.00% <nose.suite.ContextSuite context=Test13_DetectApertTor>:setup: 0.0390s
[success] 0.00% tests.tests01_geom.tests03_core.Test10_LensLin.test03_saveload: 0.0325s
[success] 0.00% tests.tests01_geom.tests03_core.Test09_LensTor.test03_saveload: 0.0321s
[success] 0.00% tests.tests01_geom.tests03_core.Test12_ApertLin.test02_saveload: 0.0319s
[success] 0.00% tests.tests01_geom.tests03_core.Test15_DetectLensTor.test04_calc_Sig: 0.0314s
[success] 0.00% tests.tests01_geom.tests03_core.Test16_DetectLensLin.test04_calc_Sig: 0.0313s
[success] 0.00% tests.tests01_geom.tests03_core.Test05_LOSTor.test03_saveload: 0.0290s
[success] 0.00% tests.tests01_geom.tests03_core.Test01_VesTor.test06_saveload: 0.0287s
[success] 0.00% tests.tests01_geom.tests03_core.Test06_LOSLin.test03_saveload: 0.0284s
[success] 0.00% tests.tests01_geom.tests03_core.Test03_StructTor.test03_saveload: 0.0266s
[success] 0.00% tests.tests01_geom.tests03_core.Test11_ApertTor.test02_saveload: 0.0263s
[success] 0.00% tests.tests01_geom.tests03_core.Test02_VesLin.test06_saveload: 0.0256s
[success] 0.00% tests.tests01_geom.tests03_core.Test14_DetectApertLin.test04_calc_Sig: 0.0252s
[success] 0.00% tests.tests01_geom.tests02_compute.test04_Ves_get_MeshCrossSection: 0.0235s
[success] 0.00% tests.tests01_geom.tests03_core.Test01_VesTor.test03_get_MeshCrossSection: 0.0166s
[success] 0.00% tests.tests01_geom.tests03_core.Test20_GDetectLensLin.test04_calc_SAngVect: 0.0163s
[success] 0.00% tests.tests01_geom.tests03_core.Test15_DetectLensTor.test01_refine_ConePoly: 0.0159s
[success] 0.00% tests.tests01_geom.tests03_core.Test02_VesLin.test03_get_MeshCrossSection: 0.0153s
[success] 0.00% tests.tests01_geom.tests03_core.Test15_DetectLensTor.test02_isInside: 0.0147s
[success] 0.00% tests.tests01_geom.tests03_core.Test19_GDetectLensTor.test04_calc_SAngVect: 0.0085s
[success] 0.00% tests.tests01_geom.tests03_core.Test16_DetectLensLin.test01_refine_ConePoly: 0.0085s
[success] 0.00% tests.tests01_geom.tests03_core.Test14_DetectApertLin.test01_refine_ConePoly: 0.0055s
[success] 0.00% tests.tests01_geom.tests03_core.Test15_DetectLensTor.test03_calc_SAngVect: 0.0048s
[success] 0.00% tests.tests01_geom.tests03_core.Test20_GDetectLensLin.test03_get_GLOS: 0.0048s
[success] 0.00% tests.tests01_geom.tests03_core.Test19_GDetectLensTor.test03_get_GLOS: 0.0044s
[success] 0.00% tests.tests01_geom.tests03_core.Test14_DetectApertLin.test03_calc_SAngVect: 0.0037s
[success] 0.00% tests.tests01_geom.tests03_core.Test16_DetectLensLin.test03_calc_SAngVect: 0.0033s
[success] 0.00% tests.tests01_geom.tests03_core.Test01_VesTor.test01_isInside: 0.0033s
[success] 0.00% tests.tests01_geom.tests03_core.Test20_GDetectLensLin.test02_isInside: 0.0032s
[success] 0.00% tests.tests01_geom.tests03_core.Test19_GDetectLensTor.test02_isInside: 0.0029s
[success] 0.00% tests.tests01_geom.tests03_core.Test02_VesLin.test01_isInside: 0.0026s
[success] 0.00% tests.tests01_geom.tests03_core.Test16_DetectLensLin.test02_isInside: 0.0022s
[success] 0.00% tests.tests01_geom.tests02_compute.test03_Ves_get_InsideConvexPoly: 0.0019s
[success] 0.00% tests.tests01_geom.tests03_core.Test02_VesLin.test02_InsideConvexPoly: 0.0015s
[fail] 0.00% tests.tests01_geom.tests03_core.Test04_StructLin.test01_isInside: 0.0012s
[success] 0.00% tests.tests01_geom.tests03_core.Test03_StructTor.test01_isInside: 0.0012s
[success] 0.00% tests.tests01_geom.tests03_core.Test01_VesTor.test02_InsideConvexPoly: 0.0010s
[success] 0.00% tests.tests01_geom.tests02_compute.test01_Ves_set_Poly: 0.0010s
[success] 0.00% tests.tests01_geom.tests03_core.Test14_DetectApertLin.test02_isInside: 0.0006s
[success] 0.00% tests.tests01_geom.tests02_compute.test02_Ves_isInside: 0.0002s
[success] 0.00% tests.tests01_geom.tests03_core.Test19_GDetectLensTor.test01_select: 0.0002s
[success] 0.00% tests.tests01_geom.tests03_core.Test07_GLOSTor.test01_select: 0.0002s
[success] 0.00% tests.tests01_geom.tests03_core.Test08_GLOSLin.test01_select: 0.0002s
[success] 0.00% tests.tests01_geom.tests03_core.Test20_GDetectLensLin.test01_select: 0.0002s
[fail] 0.00% tests.tests01_geom.tests03_core.Test20_GDetectLensLin.test05_calc_Sig: 0.0002s
[fail] 0.00% tests.tests01_geom.tests03_core.Test19_GDetectLensTor.test05_calc_Sig: 0.0002s
[error] 0.00% nose.failure.Failure.runTest: 0.0001s
[fail] 0.00% tests.tests01_geom.tests03_core.Test20_GDetectLensLin.test13_plot_Sig: 0.0001s
[fail] 0.00% tests.tests01_geom.tests03_core.Test19_GDetectLensTor.test13_plot_Sig: 0.0001s""".split('\n')

import IPY

get_ipython().system("grep -rnw 'tofu' -e 'axisbg'")

