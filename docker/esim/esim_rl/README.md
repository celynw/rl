1. `cd` to this directory first
1. Make sure your SSH agent is correctly set up

	```bash
	eval "$(ssh-agent)"
	ssh-add ~/.ssh/id_rsa
	```

1.	```bash
	docker build -t esim_rl --ssh default .
	```
