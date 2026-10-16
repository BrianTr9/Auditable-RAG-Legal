# Module loading on Bunya (sourced by setup_env.sh and _common.sh).
# RCC's Lmod needs the full name/version: a bare `module load miniforge` fails with
# "exist but cannot be loaded as requested". Resolve the newest version installed on this node
# (versions differ between CPU and GPU nodes; CUDA modules exist only on GPU nodes).
load_module() {
  local name=$1 full
  # `|| true`: with `set -e` a non-matching grep would end the script here, before the diagnostics below.
  full=$(module -t avail "$name" 2>&1 | sed 's/([^)]*)//g; s/[[:space:]]*$//' | grep -E "^${name}/" | sort -V | tail -n 1) || true
  if [[ -z $full ]]; then
    echo "No loadable '$name' module on $(hostname). 'module spider $name' says:" >&2
    module spider "$name" >&2 || true
    return 1
  fi
  echo "module load $full"
  module load "$full"
}
