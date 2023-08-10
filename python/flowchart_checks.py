import sys
import collections

import laf
from laf.fabric import LafFabric
from etcbc.preprocess import prepare
fabric = LafFabric()

version = '4b'
API = fabric.load('etcbc{}'.format(version), 'lexicon', 'valence', {
    "xmlids": {"node": False, "edge": False},
    "features": ('''
        oid otype monads
        function rela
        g_word_utf8 trailer_utf8
        lex prs sp ls vs vt nametype det gloss
        book chapter verse label number
    ''',
    '''
        mother
    '''),
    "prepare": prepare,
    "primary": False,
}, verbose='DETAIL')
exec(fabric.localnames.format(var='fabric'))

predicates = {'Pred', 'PreS', 'PreO', 'PtcO', 'PreC'}

msg('Examining verbs in predicates in clauses')
verb_dist_clause = collections.defaultdict(lambda: [])
verb_dist_clause_atom = collections.defaultdict(lambda: [])
for p in F.otype.s('phrase'):
    pf = F.function.v(p)
    if pf not in predicates: continue
    c = L.u('clause', p)
    ca = L.u('clause_atom', p)
    for w in L.d('word', p):
        if F.sp.v(w) != 'verb': continue
        verb_dist_clause[c].append(F.lex.v(w))
        verb_dist_clause_atom[ca].append(F.lex.v(w))
msg('Done')
nverbc = len(verb_dist_clause)
nverbca = len(verb_dist_clause_atom)

print('{} clauses have a predicate with a verb'.format(nverbc))
print('{} clause_atoms have a predicate with a verb'.format(nverbca))

multiples_c = 0
onelex_c = 0
for c in verb_dist_clause:
    lexes = verb_dist_clause[c]
    if len(lexes) == 1: continue
    multiples_c += 1
    if len(set(lexes)) == 1: onelex_c += 1
print('{} clauses have multiple verb occurrences of which {} are still single lexeme'.format(
        multiples_c, onelex_c,
))
multiples_ca = 0
onelex_ca = 0
for ca in verb_dist_clause_atom:
    lexes = verb_dist_clause_atom[ca]
    if len(lexes) == 1: continue
    multiples_ca += 1
    if len(set(lexes)) == 1: onelex_ca += 1
print('{} clause_atoms have multiple verb occurrences of which {} are still single lexeme'.format(
        multiples_ca, onelex_ca,
))

msg('Examining verbs in predicates in clauses (revisited)')

def known_case(vs):
    be = {'HJH[', 'HWH['}
    if len(vs) != 2: return False
    ps = collections.defaultdict(lambda: set())
    for (p,w) in vs: ps[F.function.v(p)].add(F.lex.v(w))
    if len(ps['Pred'] | ps['PreS']) == 1 and len(be & (ps['Pred'] | ps['PreS'])) != 0: return True
    return False

verb_dist = collections.defaultdict(lambda: [])
for p in F.otype.s('phrase'):
    pf = F.function.v(p)
    if pf not in predicates: continue
    c = L.u('clause', p)
    for w in L.d('word', p):
        if F.sp.v(w) != 'verb': continue
        verb_dist[c].append((p, w))
msg('Done')

of = outfile('verb_dist.txt')
multiple = 0
good = 0
for c in verb_dist:
    vs = verb_dist[c]
    if len(vs) == 1: continue
    multiple += 1
    if known_case(vs):
        good += 1
        continue
    of.write('\n{} {}:{}#{}_{}\n'.format(
        F.book.v(L.u('book', c)),
        F.chapter.v(L.u('chapter', c)),
        F.verse.v(L.u('verse', c)),
        F.number.v(L.u('sentence', c)),
        F.number.v(c),
    ))
    for (p,w) in vs:
        of.write('\t{} {} has {}\n'.format(p, F.function.v(p), F.lex.v(w)))
of.close()

msg('''
{:>5} single verb clauses
{:>5} known multiple cases
{:>5} unknown multiple verb clauses
{:>5} clauses in total'''.format(
    len(verb_dist) - multiple,
    good,
    multiple - good,
    len(verb_dist),
))

target_lexemes = {'NTN[', 'FJM[', 'BR>['}
msg('Examining the occurrences of {}'.format(', '.join(sorted(target_lexemes))))
qal_dist_all = collections.Counter()
qal_dist = collections.defaultdict(lambda: collections.Counter())
for w in F.otype.s('word'):
    lex = F.lex.v(w)
    if lex in target_lexemes and F.vs.v(w) == 'qal':
        wt = F.vt.v(w)
        wf = F.function.v(L.u('phrase', w))
        qal_dist_all[(wf,wt)] += 1
        qal_dist[lex][(wf,wt)] += 1
msg('Done')
tot = 0
for (label, n) in sorted(qal_dist_all.items(), key=lambda y: (-y[1], y[0])):
    tot += n
    print('{:<4} {:<4} {:>5} x'.format(label[0], label[1], n))
print('Total     {:>5} x'.format(tot))
for lx in sorted(qal_dist):
    print(lx)
    tot = 0
    for (label, n) in sorted(qal_dist[lx].items(), key=lambda y: (-y[1], y[0])):
        tot += n
        print('     {:<4} {:<4} {:>5} x'.format(label[0], label[1], n))
    print('     Total     {:>5} x'.format(tot))

msg('Exploring >T prefixes')
etprefixes = collections.Counter()
etprefix_words = collections.Counter()
for p in F.otype.s('phrase'):
    if F.function.v(p) != 'Objc': continue
    prefix = []
    for w in L.d('word', p):
        if F.lex.v(w) == '>T':
            found = True
            break
        prefix.append(w)
    if found:
        prefstr = '-'.join(F.sp.v(w) for w in prefix)
        etprefixes[prefstr] += 1
        for w in prefix:
            etprefix_words[F.lex.v(w)]+= 1 
msg('Done')
for x in sorted(etprefix_words.items(), key=lambda y: (-y[1], y[0]))[0:10]:
    print('{:<20}: {:>5} x'.format(x[0], x[1]))

msg('Exploring the mothers of Objc-clauses')
noc = 0
nc = 0
mothers = collections.Counter()
has_mothers = collections.Counter()
for c in F.otype.s('clause'):
    nc += 1
    if F.rela.v(c) == 'Objc':
        noc += 1
        nmothers = 0
        for x in C.mother.v(c):
            nmothers += 1
            motype = F.otype.v(x)
            mytype = motype
            if motype == 'phrase':
                mytype = 'phrase {}'.format(F.function.v(x))
            elif motype == 'clause':
                mytype = 'clause {}'.format(F.rela.v(x))
            mothers[mytype] += 1
        has_mothers[nmothers] += 1
msg('Done')
print('{} object clauses of total {}'.format(noc, nc))
totaln = 0
for otp in sorted(mothers):
    thisn = mothers[otp]
    totaln += thisn
    print('{:<16}: {:>4}x'.format(otp, thisn))
print('Total {} mothers'.format(totaln))
totaln = 0
for x in sorted(has_mothers, reverse=True):
    thisn = has_mothers[x]
    if x != 0: totaln += thisn
    print('# mothers = {:>2} for {:>4} object clauses'.format(x, thisn))
print('Total {} object clauses with a mother'.format(totaln))

msg('Investigating promotion')
predicates = {'Pred', 'PreS', 'PreO', 'PtcO', 'PreC'}
objectsf = {'Objc', 'PreO', 'PtcO'}
no_prs = {'absent', 'n/a'}
prom_preps = {'K', 'L'}

of = outfile('promotion.csv')
fields = '''book chapter verse sentence clause verbs #objs #cands'''.strip().split()
of_fmt = '{}'+('\t{}' * (len(fields)-1))+'\n'
of.write(of_fmt.format(*fields))

ncands = collections.Counter()
nobjs = collections.Counter()
nclauses = 0

for c in F.otype.s('clause'):
    nclauses += 1
    verbs = []
    cws = L.d('word', c)
    cw1 = cws[0]
    for w in cws:
        if F.sp.v(w) == 'verb':
            verbs.append(F.lex.v(w))
    cands = []
    ps = L.d('phrase', c)
    for p in ps:
        if F.function.v(p) != 'Cmpl': continue
        ws = L.d('word', p)
        w_one = ws[0]
        w_lex = F.lex.v(w_one)
        w_prs = F.prs.v(w_one)
        if w_prs not in no_prs: continue
        if w_lex not in prom_preps: continue
        cands.append(p)
    nc = len(cands)
    if nc == 0: continue

    ncands[nc] += 1

    objects = []
    for p in ps:
        if F.function.v(p) in objectsf:
            objects.append(p)
    no = len(objects)
    if no != 0:
        nobjs[no] += 1

    of.write(of_fmt.format(
        F.book.v(L.u('book', cw1)),
        F.chapter.v(L.u('chapter', cw1)),
        F.verse.v(L.u('verse', cw1)),
        F.number.v(L.u('sentence', cw1)),
        F.number.v(L.u('clause', cw1)),
        ' '.join(verbs),
        no,
        nc,
    ))
of.close()
msg('Done')
print('{:<40}: {:>6}'.format('Total clauses', nclauses))
print('{:<40}: {:>6}'.format('with any candidates', sum(ncands.values())))
for nc in sorted(ncands, reverse=True):
    print('{:<40}: {:>6}'.format('with {:>2} candidates'.format(nc), ncands[nc]))
print('{:<40}: {:>6}'.format('with any objects', sum(nobjs.values())))
for no in sorted(nobjs, reverse=True):
    print('{:<40}: {:>6}'.format('with {:>2} objects'.format(no), nobjs[no]))

msg('Investigating relative clauses')
ashers = list(F.lex.s('>CR'))
msg('The word >CR occurs {} times'.format(len(ashers)))

aclauses = collections.OrderedDict()
multiples = collections.Counter()
for a in ashers:
    c = L.u('clause', a)
    m = list(C.mother.v(c))
    if m:
        if c in aclauses:
            multiples[c]  += 1
        else:
            aclauses[c] = m[0]

msg('There are {} ashers with a mother; {} have multiple mothers'.format(len(aclauses), len(multiples)))         
for (c, n) in sorted(multiples.items(), key=lambda y: -y[1]):
    print('Clause {} has {} ashers'.format(c, n))

mothertypes = collections.defaultdict(lambda: [])
for c in aclauses:
    mothertypes[F.otype.v(aclauses[c])].append(c)
    
for (mt, ms) in sorted(mothertypes.items(), key=lambda y: -len(y[1])):
    print('{:>5} clauses have mother of type {}'.format(len(ms), mt))

of = outfile('asher.txt')
for mtype in mothertypes:
    of.write('[{}]\n'.format(mtype))
    for c in mothertypes[mtype]:
        m = aclauses[c]
        pair = sorted([c, m], key=NK)
        w = L.d('word', c)[0]
        of.write('{:<20} {:>3}:{:>3}#{:>2}_{:>2} {:<6} {}\n'.format(
            F.book.v(L.u('book', w)),
            F.chapter.v(L.u('chapter', w)),
            F.verse.v(L.u('verse', w)),
            F.number.v(L.u('sentence', w)),
            F.number.v(c),
            F.otype.v(m),
            ''.join('C' if x == c else 'M' for x in pair),
        ))
of.close()

cands = collections.defaultdict(lambda: collections.Counter())
verbs = collections.Counter()
glosses = {}
for w in F.otype.s('word'):
    ls = F.ls.v(w)
    sp = F.sp.v(w)
    if sp == 'subs':
        cands[ls][F.lex.v(w)] += 1
    elif sp == 'verb':
        verbs[F.lex.v(w)] += 1
    glosses[F.lex.v(w)] = F.gloss.v(w)

of = outfile('words.txt')
for ls in cands:
    of.write('[{}]\n'.format(ls))
    for cand in sorted(cands[ls]):
        of.write('{:<10} : {:>5} x {}\n'.format(cand, cands[ls][cand], glosses[cand]))
    of.write('\n')
of.close()
of = outfile('verbs.txt')
for vb in sorted(verbs):
    of.write('{:<10} : {:>5} x {}\n'.format(vb, verbs[vb], glosses[vb]))
of.write('\n')
of.close()



