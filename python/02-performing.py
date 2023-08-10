import ctcsound
cs = ctcsound.Csound()

ret = cs.compile_("csound", "-o", "dac", "examples/02-a.orc", "examples/02-a.sco")
if ret == ctcsound.CSOUND_SUCCESS:
    cs.perform()
    cs.reset()

ret = cs.compile_("csound", "examples/02-a.csd")
if ret == ctcsound.CSOUND_SUCCESS:
    cs.perform()
    cs.reset()

ret = cs.compileCsd("examples/02-a.csd")
if ret == ctcsound.CSOUND_SUCCESS:
    cs.start()
    cs.perform()
    cs.reset()

csd = '''
<CsoundSynthesizer>

<CsOptions>
  -d -o dac -m0
</CsOptions>

<CsInstruments>
sr     = 48000
ksmps  = 100
nchnls = 2
0dbfs  = 1

          instr 1
idur      =         p3
iamp      =         p4
icps      =         cpspch(p5)
irise     =         p6
idec      =         p7
ipan      =         p8

kenv      linen     iamp, irise, idur, idec
kenv      =         kenv*kenv
asig      poscil    kenv, icps
a1, a2    pan2      asig, ipan
          outs      a1, a2
          endin
</CsInstruments>

<CsScore>
i 1 0 1 0.5 8.06 0.05 0.3 0.5
e 1.5
</CsScore>
</CsoundSynthesizer>
'''
ret = cs.compileCsdText(csd)
if ret == ctcsound.CSOUND_SUCCESS:
    cs.start()
    cs.perform()
    cs.reset()

ret = cs.compile_("csound", "examples/02-a.csd")
if ret == ctcsound.CSOUND_SUCCESS:
    while not cs.performKsmps():
        print('.', end='')
    print()
    cs.reset()

ret = cs.compile_("csound", "examples/02-a.csd")
if ret == ctcsound.CSOUND_SUCCESS:
    while not cs.performBuffer():
        print('.', end='')
    print()
    cs.reset()

