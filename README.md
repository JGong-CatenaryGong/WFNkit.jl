# WFNkit.jl

The wavefunction toolkit, WFNkit for short, is a *developing* package of Julia modules for dealing with wavefunctions generated by quantum mechanics calculation software suites like Gaussian, ORCA, etc. The package is mainly inspired by the venerable [multiwfn](http://sobereva.com/multiwfn/), developed by Sobreva at [Beijing Kein Research Center for Natural Sciences](http://www.keinsci.com/) and many other predecessors.

WFNkit.jl is not mean to be the competitor to multiwfn, but rather wants to provide simple and dynamic code modules that are easy to embed and reuse to researchers.

## Developed modules

There are developed modules in WFNkit.jl:

- coords.jl: For extracting molecular geometries from xyz files;
- wfreader.jl: For extracting wavefunction parameters from wfn files;
- wfreader_thread.jl: Threaded version of wfreader.jl;
- wfcalc.jl: For calculating value of real space functions (wavefunction, electronic density, gradient of electronic density) from extracted wavefunction parameters;
- wfcalc_opt.jl: Threaded and CUDA version of wfcalc.jl.

## Developing modules

The following module is under construction:

- molread.jl: For reading SMILES.