For file read/write and get/set, I designed

- [Params], to make runtime config get/set/load/dump easily,
- [globs], a global/local/runtime environment variables manager.
- [Saver], to help you save/load/manage your checkpoints/models in one class.

For data processing, I designed

- [Builder], to hold nearly all dataset formats and special operations by one class,
- [Delegate], a torch dataset-like class to hold other dataset formats and operations that builder can't.

For managing experiments, I designed

- [Experiment], which can
    - make you build a suitable directory and file path in one place,
    - make you record lightweight data, and
    - help you make snapshot for your project code (based on git), which can make each result recoverable and
      reproducible
- [random manager], a cross-lib(random/numpy/pytorch) random seed manager

For log and meter variables produced during experiment, I designed

- [Meter] to meter every thing in appropriate format, and
- [Logger] to log every thing in appropriate format.

Finally, I designed [Trainer] to bundle all module above for deep learning experiment.


As you can see, These modules covered most demandings on deeplearning 

You can find what you want and click the link to quickly learn HOW TO USE it! All module is designed easy to use, it's
my principles.


