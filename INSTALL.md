# How to install and configure required tools?

## Required development tools

### A. Quick start for quick feature discovery

Therefore, the quickest and most minimal way to get touch is to install mise on your system and prepare tools with mise (but very limited, by speed and by features):

1. `brew install mise && brew install txn2/tap/kubefwd && mise install uv@latest && mise use -g uv@latest`

That's all, go to **Shell configuration** section.

### B. Engineer full-featured setup

**True engineer way**: prepare rust environment, build mise and configure shell:

1. Install Cargo via [Rustup](https://doc.rust-lang.org/book/ch01-01-installation.html).

2. Do not forget add cargo path for your shell (and add it to your shell):
    - `export PATH="~/.cargo/bin:$PATH"`

3. Compile sccache to avoid electricity bills (modern ccache drop-in replacement tool which blazing fast and speed-up any compilation):
    - `cargo install sccache`

4. Activate sccache (also add it to your shell):

    - `export RUSTC_WRAPPER="~/.cargo/bin/sccache"`
    - and you can prepare your shell for another tools, clang and fortran for example
    - `export CC="sccache clang"`
    - `export FC="sccache gfortran"`
    - `export CXX="sccache clang++"`

5. Deploy cargo packages updater and mise:
    - `cargo install cargo-update mise`

6. That's all, you have last version of optimized tools; for update all packages just run sometime:
    - `rustup update && cargo install-update --all`

7. If you're developer and want contribute to project, also immediately install kubefwd:
    - `brew install txn2/tap/kubefwd`

8. Build current Astral UV (development-last-versioon not required, mise install uv too and handle releases of):
    - `mise install uv@latest && mise use -g uv@latest`

## Shell integration

1. **Mise provide dotenv functionality** (automatic read per-project environment variables from .env file and from mise.toml config) from the box with batteries, but it required entry hook in your shell. Add the following lines to your `~/.zshrc` file (or the corresponding file for your shell, e.g., `~/.bashrc` for bash):
    - `eval "$(mise activate zsh)"`

2. Also you can want using autocompletion, same story for zsh:
    - `eval "$(mise completion zsh && uv generate-shell-completion zsh)"`

3. If you use direnv for more complex environment management, you can also add the following line to your `~/.zshrc` file:
    - `eval "$(mise activate direnv)"`

4. Restart your shell session:
    - `exec zsh`

5. Note: If you modify the `.envrc` file, `direnv` will not automatically reload it. You need to leave and re-enter the project directory for the changes to be applied: `cd .. && cd -`.

## How to use the project?

### Project contribution in action

1. (if applicable) Authenticate with Google Cloud, because we work with services deployed in kubernetes cluster:
    - `gcloud auth login --update-adc`

2. Go to project root, install environment by `make install`, put your settings (gitlab token, zitadel secrets, used staging environment, etc) to .env file and rerun `make install` till you have succeeded setup).

3. Run port-forwarding from cluster services to your machine:

    - `make ports`
    - this command will start **kubefwd** and forward ports from all services defined in **justfile** to your local ports (really!), you can change it, but by default it will forward temporal, zitadel, utils services and entire scope from **STAGING** environment variable (by default it's **example**), you can change in your .env file
    - sudo required by used `kubefwd` for temporary hostnames aliasing in /etc/hosts, read attached **justfile** for details

4. That's all, run the application in development mode (with reloading and debug headers):
    - `make develop`

5. If you need fine-grained development control, just read your .env file.

6. Also uv everytime install by default local environment with _development_ dependencies group which include simple [better-exceptions](https://github.com/qix-/better-exceptions) library. This library enabled by default for development (and do not included to production packaging, sic!), but you can simple turn it off by **BETTER_EXCEPTIONS** environment variable setted to 0 in your .env.

## Work with project dependencies

### Warning about pip

**NEVER CALL pip, NEVER!** Instead it use native uv calls, [read uv manual](https://docs.astral.sh/uv/guides/projects/#managing-dependencies), it's very easy, for example:

1. Set or change python version, before run do not forget change your python version in pyproject.toml:
    - `uv python pin 3.13 && uv sync`

2. If python 3.13 isn't installed, run `mise install python@3.13` and mise download and install python 3.13, recreate virtual environment with 3.13 context. Do not forget to pin python version by uv from previous step to **mise.toml**(and, may be you need to update your **pyproject.toml**).

3. Just add new dependency:
    - `uv add phpbb<=1.2`

4. Add some development library:
    - `uv add --group development backyard-memleak`

5. Work with locally cloned repository:
    - `uv add --editable ~/src/lib/chrome-v8-core`

6. if you work on service (not library), you must lock dependencies graph state to uv.lock before commit:
    - `uv lock`

When you need to restore your environment after any experiments, just run `make install`, **feel free, it's spent only few seconds** to restore your environment to absolutely clean state.

### Project workflow commands

(read **justfile** for details)

1. Your main command is `make help`, feel free to run it, it will show you all available commands and their descriptions.

2. `make install`:

    - if you run this in project first time, all packaging command failed because we need to install internal corporate packages; don't worry, it's easy, just open for edit .env file and put to **UV_INDEX_PRIVATE_PASSWORD** your gitlab deployment (read only) token and rerun `make install`

3. `make ports`:
    - start kubefwd and forward ports from all services defined in **justfile** to yout local machine with reacheble and human-readable hostnames

4. `make develop`:
    - run development server with application

5. `make test`:
    - run tests after work

6. `make pre-commit`:
    - run all compatible linters and checkers with --fix and --edit mode, be careful, it can change your code, so always commit all changes before run it! (and do not forget to add all changes after run it, because some files can be unstaged after this command); use it for manual run all checks

7. `make dock`:
    - builds production ready docker image prepared for publishing

8. `uv sync`
    - read project dependencies graph from pyproject.toml and install it to virtual environment

9. `uv lock`:
    - dump dependencies graph state to uv.lock and lock dependecies

10. `make publish`:
    - builds production ready docker image and publish it to registry, read **justfile** for details

11. `make upgrade`:
    - read project dependencies graph from pyproject.toml
    - fetch information about all updated packages, recreate dependencies graph and install it to virtual environment by `uv sync --upgrade`
    - update `uv.lock` with updated packages version by `uv lock --upgrade`
    - show all installed packages in local virtual environment
    - all you need it's just manually update versions in pyproject.toml after visual review
