from onyx.adaptive_controller import AdaptiveGammaConfig, AdaptiveGammaController


def make_controller(initial=4):
    return AdaptiveGammaController(
        AdaptiveGammaConfig(
            min_gamma=1,
            max_gamma=8,
            initial_gamma=initial,
            window_size=2,
        )
    )


def test_high_acceptance_with_verify_bottleneck_increases_gamma():
    controller = make_controller(initial=4)

    next_gamma = controller.observe(
        proposed=4,
        accepted=4,
        draft_time=0.01,
        verify_time=0.03,
        mask_time=0.0,
    )

    assert next_gamma == 5
    assert controller.adjustments == 1


def test_low_acceptance_decreases_gamma():
    controller = make_controller(initial=4)

    next_gamma = controller.observe(
        proposed=4,
        accepted=1,
        draft_time=0.01,
        verify_time=0.03,
        mask_time=0.0,
    )

    assert next_gamma == 3
    assert controller.adjustments == 1


def test_high_mask_overhead_decreases_gamma():
    controller = make_controller(initial=4)

    next_gamma = controller.observe(
        proposed=4,
        accepted=4,
        draft_time=0.01,
        verify_time=0.03,
        mask_time=0.03,
    )

    assert next_gamma == 3
    assert controller.adjustments == 1


def test_gamma_stays_within_bounds():
    controller = make_controller(initial=1)

    for _ in range(4):
        gamma = controller.observe(
            proposed=4,
            accepted=0,
            draft_time=0.01,
            verify_time=0.01,
            mask_time=0.0,
        )

    assert gamma == 1

    controller = make_controller(initial=8)
    for _ in range(4):
        gamma = controller.observe(
            proposed=4,
            accepted=4,
            draft_time=0.01,
            verify_time=0.03,
            mask_time=0.0,
        )

    assert gamma == 8

