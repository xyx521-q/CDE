import unittest

import numpy as np

from tools.run_microgrid_milp_oracle import apply_parameter_overrides, solve_dispatch


class MicrogridMilpOracleTest(unittest.TestCase):
    def test_parameter_overrides_do_not_mutate_original_params(self):
        params = {
            "dg_max": np.array([2.0, 2.0, 2.0]),
            "battery_caps": np.array([5.0, 5.0, 5.0]),
        }

        overridden = apply_parameter_overrides(
            params,
            dg_max_override=80.0,
            battery_cap_override=200.0,
        )

        np.testing.assert_allclose(overridden["dg_max"], [80.0, 80.0, 80.0])
        np.testing.assert_allclose(overridden["battery_caps"], [200.0, 200.0, 200.0])
        np.testing.assert_allclose(params["dg_max"], [2.0, 2.0, 2.0])
        np.testing.assert_allclose(params["battery_caps"], [5.0, 5.0, 5.0])

    def test_battery_shifts_energy_and_returns_to_initial_soc(self):
        params = {
            "dg_max": np.array([0.0]),
            "battery_caps": np.array([10.0]),
            "raw_ch": np.array([1.0]),
            "raw_dis": np.array([1.0]),
            "costb": np.array([0.0]),
            "env_param": np.array([0.0]),
        }
        result = solve_dispatch(
            load_pv=np.array([5.0, 5.0]),
            price=np.array([1.0, 10.0]),
            params=params,
            initial_soc=0.5,
            terminal_soc=0.5,
            cycle_cost=0.0,
        )

        self.assertTrue(result["success"], result["message"])
        self.assertAlmostEqual(result["soc"][0, 0], 0.5)
        self.assertAlmostEqual(result["soc"][-1, 0], 0.5)
        self.assertAlmostEqual(result["charge"][0, 0], 3.0, places=6)
        self.assertAlmostEqual(result["discharge"][1, 0], 3.0, places=6)
        self.assertLess(result["grid_purchase_cost"], 55.0)


if __name__ == "__main__":
    unittest.main()
