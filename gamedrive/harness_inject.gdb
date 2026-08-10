set pagination off
set confirm off
set print thread-events off
set unwind-on-signal on
handle SIGSEGV stop print nopass
handle SIGFPE stop print nopass

# Use one main-loop stop to start the asynchronous native load.  The second
# breakpoint is conditional, so the game can run the expensive native turn
# worker without ptrace-stopping the GUI thread on every frame.
break mainLoop
commands
  silent
  printf "== native probe: mainLoop reached; starting load ==\n"
  set $ignored = (int)d3_start_load()
  disable 1
  enable 2
  continue
end

# mainLoop is one long-lived internal loop, so it is not re-entered once the
# process starts.  GameProc is called from the loop once per active frame and
# is therefore the recurring main-thread boundary needed by the probe.
break *0x625653
condition 2 (*(unsigned char*)0xa3a9f8 != 0 && *(unsigned char*)0xa15c50 != 0)
commands
  silent
  printf "== native probe: load/turn completion boundary ==\n"
  set $probe = (int)d3_maybe_run()
  if $probe != 0
    printf "== native probe: load/turn/save completed ==\n"
    quit
  end
  continue
end

disable 2

run -silent
