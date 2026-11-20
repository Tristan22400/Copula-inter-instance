#!/bin/bash
# Holder job: keeps a GPU node reserved; work runs on it through oarsh.
#OAR -n hold-npbench
sleep "${1:-43000}"
