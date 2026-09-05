Output docs/assets/tinyexp-demo.gif

Set Shell "zsh"
Set FontSize 22
Set Width 1280
Set Height 720
Set TypingSpeed 80
Set CursorBlink false
Set Theme "Catppuccin Mocha"

Type "clear"
Enter

Type "echo 'TinyExp: your configured experiment is your entrypoint'"
Enter
Sleep 2s

Type "python -m tinyexp.examples.mnist_exp mode=help"
Enter
Sleep 8s

Type "python -m tinyexp.examples.mnist_exp mode=help dataloader_cfg.train_batch_size_per_device=16 ray_cfg.ray_num_worker=1"
Enter
Sleep 8s

Type "echo 'One experiment file. Structured config. CLI overrides.'"
Enter
Sleep 3s
