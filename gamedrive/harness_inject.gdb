set pagination off
set confirm off
set print thread-events off
set unwind-on-signal on
handle SIGSEGV stop print nopass
handle SIGFPE stop print nopass

# Use one main-loop stop to start the asynchronous native load.  The second
# breakpoint is enabled only after that call and runs the lightweight probe
# check once per active frame until the native worker starts.
break mainLoop
commands
  silent
  printf "== native probe: mainLoop reached; starting load ==\n"
  set $ignored = (int)d3_start_load()
  disable 1
  enable 2
  enable 3
  continue
end

# mainLoop is one long-lived internal loop, so it is not re-entered once the
# process starts.  GameProc is called from the active path; the SDL_Delay call
# site is called from the inactive/unfocused path.  Together they cover both
# post-load states without stopping on every internal loop iteration.
break *0x625653
condition 2 (*(unsigned char*)0xa3a9f8 != 0 && *(unsigned char*)0xa15c51 != 0)
commands
  silent
  set $probe = (int)d3_maybe_run()
  if $probe != 0
    printf "== native probe: load/turn/save completed ==\n"
    quit
  end
  continue
end

break *0x625430
condition 3 (*(unsigned char*)0xa3a9f8 != 0 && *(unsigned char*)0xa15c51 != 0)
commands
  silent
  set $probe = (int)d3_maybe_run()
  if $probe != 0
    printf "== native probe: load/turn/save completed ==\n"
    quit
  end
  continue
end

disable 2
disable 3

run -silent
