Output docs/assets/tinyexp-demo-short.gif

Set Shell "zsh"
Set FontSize 24
Set Width 1280
Set Height 720
Set TypingSpeed 35ms
Set CursorBlink false
Set Theme "Catppuccin Mocha"

Type "clear"
Enter

Type "echo 'TinyExp: your configured experiment is your entrypoint'"
Enter
Sleep 1s

Type "uv run python tinyexp/examples/mnist_exp.py mode=help"
Enter
Sleep 4s

Type "uv run python tinyexp/examples/mnist_exp.py mode=help dataloader_cfg.train_batch_size_per_device=16 num_worker=1"
Enter
Sleep 5s

Type "echo 'Simple experiment management, no framework ceremony.'"
Enter
Sleep 2s
