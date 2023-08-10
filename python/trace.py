get_ipython().magic('reload_ext pytriqs.magic')

get_ipython().run_cell_magic('triqs', '-u stl', '#include <chrono>\n#include <iostream>\n\nstd::pair<std::vector<int>,array<double,2>> trace_time(int reps){\n\n using namespace triqs;\n using namespace triqs::arrays;\n\n std::vector<int> size;\n array<double,2> times(3,100);\n\n for(int i=0, dim=10; i<100; i++, dim+=10){\n\n  auto A = matrix<double>(dim, dim); A()=0;\n  auto B = matrix<double>(dim, dim); B()=0;\n\n  double time1=0, time2=0, time3=0;\n\n  std::chrono::system_clock::time_point start = std::chrono::system_clock::now();\n  for(int r=0; r<reps; ++r) auto tr = trace(A+B); // Uses expression template\n  std::chrono::duration<double> sec = std::chrono::system_clock::now() - start;\n  time1 += sec.count();\n \n  start = std::chrono::system_clock::now();\n  for(int r=0; r<reps; ++r) auto tr = trace(matrix<double>(A+B)); // Object oriented way\n  sec = std::chrono::system_clock::now() - start;\n  time2 += sec.count();\n\n  start = std::chrono::system_clock::now();\n  double tr;\n  for(int r=0; r<reps; ++r){\n   tr = 0.;\n   for (int ind=0; ind<dim; ind++) // Low level\n    tr += A(ind,ind)+B(ind,ind);\n  }\n  sec = std::chrono::system_clock::now() - start;\n  time3 += sec.count();\n\n  std::cerr << tr; // Reuse tr, otherwise optimizer eliminates the loop \n\n  size.push_back(dim);\n  times(0,i)=time1/reps;\n  times(1,i)=time2/reps;\n  times(2,i)=time3/reps;\n }\n\n return std::make_pair(size,times);\n\n}')

s,times = trace_time(1000)

figure(figsize=(15,5))
gs=GridSpec(1,2)
subplot(gs[0])
xlim(-0.0001,500)
ylim(0,0.001)
xlabel('matrix size')
ylabel('t (ms)')
plot(s,times[0],label='TRIQS', lw=4.0)
plot(s,times[1],label='matrix(A+B)', lw=2.0)
plot(s,times[2],label='low level', lw=4.0)
legend()
subplot(gs[1])
xlim(0,500)
ylim(0,0.00001)
xlabel('matrix size')
ylabel('t (ms)')
plot(s,times[0],label='TRIQS', lw=3.0)
plot(s,times[1],label='matrix(A+B)', lw=2.0)
plot(s,times[2],label='low level', lw=3.0)
legend()
savefig('trace.pdf')



