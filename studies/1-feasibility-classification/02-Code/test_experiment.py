"""Numerical and scientific invariants, independent of the pilot outcomes."""
import copy
import json
from pathlib import Path
import unittest

import experiment as e

e.resource_guard(e.read(e.ROOT / "03-Data/config/execution-v1.json"), e.ROOT)
e.initialize()
torch, np = e.torch, e.np


def optimizer_digest(opt):
    import hashlib
    digest = hashlib.sha256()
    def visit(value):
        if isinstance(value, dict):
            for key in sorted(value, key=str):
                digest.update(str(key).encode())
                visit(value[key])
        elif isinstance(value, (list, tuple)):
            for v in value:
                visit(v)
        elif isinstance(value, torch.Tensor):
            digest.update(value.numpy().tobytes())
        else:
            digest.update(str(value).encode())
    visit(opt.state_dict())
    return digest.hexdigest()


class ExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = e.read(e.ROOT / "03-Data/prepared/task-families.json")
        cls.task = cls.tasks[0]
        cls.config = e.read(e.ROOT / "03-Data/config/execution-v1.json")

    def test_independent_labels_and_encoding(self):
        for task in self.tasks:
            for env, data in task["environments"].items():
                rows = data["examples"]
                x, y = e.encode(rows)
                expected = []
                for i, r in enumerate(rows):
                    f, g = task["f_tables"][data["f_version"]], task["g_tables"][data["g_version"]]
                    target = f[r["a"]] if r["query"] == "f" else g[r["b"]] if r["query"] == "g" else (f[r["a"]] + g[r["b"]]) % 8
                    self.assertEqual(target, r["target"])
                    vector = [0.] * 19
                    vector[["f", "g", "combine"].index(r["query"])] = 1.
                    if r["a"] is not None:
                        vector[3+r["a"]] = 1.
                    if r["b"] is not None:
                        vector[11+r["b"]] = 1.
                    expected.append(vector)
                np.testing.assert_array_equal(x.numpy(), expected)
                np.testing.assert_array_equal(y.numpy(), [r["target"] for r in rows])

    def test_numpy_forward(self):
        model = e.network(99001)
        rows = self.task["environments"]["E00"]["examples"]
        x = e.encode(rows)[0].numpy().astype(np.float64)
        for layer in (0, 2, 4):
            x = x @ model[layer].weight.detach().numpy().astype(np.float64).T + model[layer].bias.detach().numpy()
            if layer != 4:
                x = np.maximum(x, 0)
        np.testing.assert_allclose(e.predict(model, rows), x, rtol=2e-5, atol=2e-7)

    def test_input_has_no_labels_or_environment(self):
        rows = self.task["environments"]["E00"]["examples"]
        other = copy.deepcopy(rows)
        for r in other:
            r["target"] = (r["target"] + 1) % 8
            r["id"] = "changed environment"
        np.testing.assert_array_equal(e.encode(rows)[0], e.encode(other)[0])

    def test_only_changed_primitive_reaches_update(self):
        for branch, primitive, stable in (("update_f", "f", "g"), ("update_g", "g", "f")):
            batch = e.update_batch(self.task, branch, "changed_primitive")
            self.assertEqual(len(batch), 8)
            self.assertTrue(all(r["query"] == primitive for r in batch))
            poisoned = copy.deepcopy(self.task)
            for r in poisoned["environments"][branch]["examples"]:
                if r["query"] != primitive:
                    r["target"] = (r["target"] + 1) % 8
            self.assertEqual(batch, e.update_batch(poisoned, branch, "changed_primitive"))
            old = [r["target"] for r in self.task["environments"]["E00"]["examples"] if r["query"] == stable]
            new = [r["target"] for r in self.task["environments"][branch]["examples"] if r["query"] == stable]
            self.assertEqual(old, new)
        self.assertEqual(len(e.update_batch(self.task, "update_f", "old_environment")), 64)
        self.assertEqual(e.update_batch(self.task, "update_f", "no_update"), [])

    def test_branch_and_optimizer_isolation(self):
        model = e.network(99002)
        opt = torch.optim.Adam(model.parameters(), lr=.003)
        x, y = e.encode(e.update_batch(self.task, "update_f", "old_primitive"))
        for _ in range(3):
            e.step(model, opt, x, y)
        before_model, before_opt = e.state_hash(model), optimizer_digest(opt)
        retained, retained_opt = e.clone_branch(model, opt, self.config, "retained")
        reset, reset_opt = e.clone_branch(model, opt, self.config, "reset")
        self.assertEqual(optimizer_digest(retained_opt), before_opt)
        self.assertEqual(len(reset_opt.state), 0)
        e.step(retained, retained_opt, x, y)
        e.step(reset, reset_opt, x, y)
        self.assertEqual(e.state_hash(model), before_model)
        self.assertEqual(optimizer_digest(opt), before_opt)
        self.assertTrue(all(int(v["step"]) == 4 for v in retained_opt.state.values()))
        self.assertTrue(all(int(v["step"]) == 1 for v in reset_opt.state.values()))

    def test_no_update_exact_identity(self):
        model = e.network(99003)
        opt = torch.optim.Adam(model.parameters(), lr=.003)
        rows = self.task["environments"]["update_f"]["examples"]
        clone, no_opt = e.clone_branch(model, opt, self.config, "none")
        self.assertIsNone(no_opt)
        for _ in range(7):
            np.testing.assert_array_equal(e.predict(model, rows), e.predict(clone, rows))
        self.assertEqual(e.state_hash(model), e.state_hash(clone))

    def test_deterministic_updates_and_paired_initialization(self):
        x, y = e.encode(e.update_batch(self.task, "update_g", "old_primitive"))
        models = [e.network(99004), e.network(99004)]
        self.assertEqual(e.state_hash(models[0]), e.state_hash(models[1]))
        for model in models:
            opt = torch.optim.Adam(model.parameters(), lr=.003)
            for _ in range(12):
                e.step(model, opt, x, y)
        self.assertEqual(e.state_hash(models[0]), e.state_hash(models[1]))

    def test_fail_closed_resource_guard(self):
        cfg = dict(self.config, minimum_available_memory_gib=1e6)
        with self.assertRaises(RuntimeError):
            e.resource_guard(cfg, e.ROOT)


if __name__ == "__main__":
    unittest.main(verbosity=2)
