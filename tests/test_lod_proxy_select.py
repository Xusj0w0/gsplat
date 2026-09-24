import unittest
import torch

from gsplat.lod import backend


class CompactProxySelectTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_matches_full_layout(self):
        for warm in (False, True):
            for tau in (3.0, 6.0, 15.0):
                with self.subTest(warm=warm, tau=tau):
                    self._check(warm, tau)

    def _check(self, warm, tau):
        blocks, leaves, mid, coarse = 3, 16, 4, 2
        total = leaves + mid + coarse
        device = torch.device("cuda")
        generator = torch.Generator(device=device).manual_seed(1234)

        geometry = torch.zeros((blocks, total, 11), device=device)
        geometry[..., :2] = torch.randn(
            (blocks, total, 2), generator=generator, device=device
        )
        geometry[..., 2] = torch.rand(
            (blocks, total), generator=generator, device=device
        ) + 4.0
        geometry[..., 3] = 0.5
        geometry[..., 4:7] = (
            torch.rand((blocks, total, 3), generator=generator, device=device)
            + 0.05
        )
        geometry[..., 7] = 1.0
        geometry = geometry.reshape(-1, 11).contiguous()
        proxy_geometry = geometry.view(blocks, total, 11)[:, leaves:].reshape(-1, 11)

        counts = torch.ones((blocks, total), dtype=torch.int32, device=device)
        counts[:, leaves : leaves + mid] = leaves // mid
        counts[:, leaves + mid :] = mid // coarse
        counts = counts.reshape(-1).contiguous()
        view = torch.eye(4, dtype=torch.float32, device=device)
        K = torch.tensor(
            [[800.0, 0.0, 400.0], [0.0, 800.0, 300.0], [0.0, 0.0, 1.0]],
            device=device,
        )

        full = backend().select(
            geometry, counts, view, K, leaves, mid, coarse, 800, 600, tau, warm
        )
        compact = backend().select_proxy(
            proxy_geometry,
            counts,
            view,
            K,
            leaves,
            mid,
            coarse,
            800,
            600,
            tau,
            warm,
        )

        for expected, actual in zip(full, compact):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
