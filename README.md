# cs224u-creativity

Course project for CS224U 

Project members: Zi Ying (Kathy) Fan, Xubo Cao

# User Guide
## Main files of interest
Our raw data is in Idea Ratings_Berg_2019_OBHDP.xlsx.

Our code is mainly found across the python files:
* dataset.py
* models.py
* training.py
* utils.py
* constants.py
* semdis.py

We also have python notebooks that demonstrate how to use this code in a full pipeline, from data processing to training and testing. The main notebooks of interest are:
* Distilbert_linear_regression.ipynb contains an example for how to run our linear classifier.
* Distilbert_rnn.ipynb contains an example for how to run our rnn classifier.

# Dev Guide
## Troubleshooting
GPU run out of memory: run `nvidia-smi` to get the PID of the current job, and run `sudo kill -9 {PID}` to clear it.

## Git
My regular flow is to run `git pull` to grab any updates, `git status` to check which files are locally modified, `git add {FILE} {FILE}` to add changes to a commit, `git commit -m "YOUR DESCRIPTION HERE"` to create the commit, and then `git push`. If you want to discard local changes completely (irreversible), you can run `git checkout -f {FILE}`. (The -f option is force).

## GCP
### Specs/How-to
Project name: cs224u-creativity

Project id: prime-script-314418

VM instance: Compute Engine -> VM instances (note: you can pin Compute Engine to your left navbar for easy access)
* Current VM: cs224u-vm-vm, with external IP address 34.145.81.99. For VM specs, see setup under https://github.com/cs231n/gcloud
* To ssh into the VM: Click on SSH dropdown -> view gcloud command. It should be something like `gcloud beta compute ssh --zone "us-west1-b" "cs224u-vm-vm"  --project "prime-script-314418"`.
* Note: if you need to edit the VM, you must stop it first

Billing: currently using Kathy's account ($300 free credits). Should be fine for awhile.

Misc:
* "Notifications" icon on upper right is helpful for indicating when an action is ready (ex finished start-up of VM, finished stopping VM, etc)
* If you run gcloud and are prompted `Generating public/private rsa key pair. Enter passphrase (empty for no passphrase):` just leave it empty.

### One-time setup
* Follow https://github.com/cs231n/gcloud "Sign up GCP for the first time" to create an account if you don't have one yet
* Under "configure your project" you do not need to create another project; but you might need to upgrade your account to the full one
* Follow the "install gcloud command-line tools" step. 
  * Try starting up the VM, running the "first time setup script" and verification underneath to see the GPU in action.
  * Unclear: I had set 'cs224u' as the Jupyter notebook password... is this the password everyone uses to access Jupyter notebooks, or is it configured individually?
  * Try running `jupyter notebook` from the cs224u-creativity directory to open any notebook. Note that you will need the external IP here. The port that the notebook connects to should be indicated in the terminal.
  * Please stop the VM after you're done trying things out.


## TODO at end of project:
* release static IP address
