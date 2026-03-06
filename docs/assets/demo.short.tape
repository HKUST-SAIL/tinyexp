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

Type "echo '$ pip install tinyexp'"
Enter
Sleep 1s

Type "echo 'from tinyexp import store_and_run_exp' > run_mnist_demo.py"
Enter
Type "echo 'from tinyexp.examples.mnist_exp import Exp' >> run_mnist_demo.py"
Enter
Type "echo >> run_mnist_demo.py"
Enter
Type "echo 'store_and_run_exp(Exp)' >> run_mnist_demo.py"
Enter
Sleep 1s

Type "cat run_mnist_demo.py"
Enter
Sleep 2s

Type "uv run python run_mnist_demo.py mode=help"
Enter
Sleep 5s

Type "uv run python run_mnist_demo.py mode=help dataloader_cfg.train_batch_size_per_device=16 num_worker=1"
Enter
Sleep 6s

Type "echo 'Configured experiment = entrypoint.'"
Enter
Sleep 2s
