def correlator_matrix_debug(
    use_pickle: bool,
    task_id:int,
    operators: List[QuantumNum],
    peram_dir,
    meson_dir,
    nt: int,
    channel: str,
    cfg_id,  # each cfg is processed one by one; maybe this can be parallelized
    ncfg: int,
    ntsrc: int,
    nvec: int,
    dry_run: bool = True,  # Enable dry-run mode
):
    # Load pickle files if specified
    if use_pickle:
        pick_light = 'peram_light_1001.pkl'
        pick_strange = 'peram_strange_1001.pkl'
        peram = pd.read_pickle(pick_light)
        peram_strange = pd.read_pickle(pick_strange)
        print(f"Loaded perambulators from pickle:")
        print(f"Light: {peram.shape}, Strange: {peram_strange.shape}")
    else:
        peram_filename = f"peram_{nvec}_cfg{cfg_id}.h5"
        for file in os.listdir(peram_dir):
            if file == peram_filename:
                peram_file = os.path.join(peram_dir, file)
                break
        peram = load_peram(peram_file, nt, nvec, ntsrc)

    # Set meson HDF5 file path
    meson_filename = f"meson-{nvec}_cfg{cfg_id}.h5"
    for file in os.listdir(meson_dir):
        if file == meson_filename:
            meson_file = os.path.join(meson_dir, file)
            break

    # Handle backward perambulator based on strangeness
    for i, op in enumerate(operators):
        if operators[op].strange != 0:
            peram_back = reverse_perambulator_time(peram_strange)
        else:
            peram_back = reverse_perambulator_time(peram)

    if not dry_run:
        # Initialize meson matrix only in non-dry-run mode
        meson_matrix = np.zeros((len(operators), len(operators), ncfg, nt), dtype=np.cdouble)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    print(f"Dry-run mode: {dry_run}")
    print(f"Configuration ID: {cfg_id}, Channel: {channel}")
    print(f"Number of configurations: {ncfg}, Number of timeslices: {nt}, Source time slices: {ntsrc}\n")

    # Iterate through operators and print what would be done
    for src_idx, (src_name, src_op) in enumerate(operators.items()):
        for snk_idx, (snk_name, snk_op) in enumerate(operators.items()):
            for tsrc in range(ntsrc):
                for t in range(nt):
                    print(f"Would process:")
                    print(f"  Source operator: {src_name} ({src_op})")
                    print(f"  Sink operator: {snk_name} ({snk_op})")
                    print(f"  Time source slice: {tsrc}, Time slice: {t}")
                    print(f"  Expected dataset path: /{src_name}_{snk_name}/tsrc_{tsrc}/cfg_{cfg_id}")

                    # Debugging shapes of arrays (mocked here)
                    print("  Shapes:")
                    print(f"    tau: {peram[tsrc, t, :, :, :, :].shape}")
                    print(f"    tau_: {peram_back[tsrc, t, :, :, :, :].shape}")

                    if src_op.deriv is None:
                        print("    Source phi_0: Local contraction")
                    elif src_op.deriv == "nabla":
                        print("    Source phi_0: Nabla contraction")
                    elif src_op.deriv in ["B", "D"]:
                        print(f"    Source phi_0: {'B' if src_op.deriv == 'B' else 'D'} contraction")

                    if snk_op.deriv is None:
                        print("    Sink phi_t: Local contraction")
                    elif snk_op.deriv == "nabla":
                        print("    Sink phi_t: Nabla contraction")
                    elif snk_op.deriv in ["B", "D"]:
                        print(f"    Sink phi_t: {'B' if snk_op.deriv == 'B' else 'D'} contraction")

                    print("-" * 80)

    print("Dry-run completed. No computations or file writes performed.")