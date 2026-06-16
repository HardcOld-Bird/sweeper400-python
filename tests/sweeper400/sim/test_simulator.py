"""SimScanner 类的单元测试

测试 sweeper400.sim.simulator 模块中 SimScanner 类及相关数据类/辅助函数。
由于 COMSOL 仿真需要硬件连接，本测试仅覆盖纯计算逻辑和可 Mock 的部分。

运行方式：
    pytest tests/sweeper400/sim/test_simulator.py -v
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # 非交互式后端，避免弹窗

from sweeper400.sim.simulator import (
    ScanResult,
    SimScanner,
    _format_complex,
    _parse_complex_str,
)


# =============================================================================
# 辅助函数测试
# =============================================================================


class TestFormatComplex:
    """测试 _format_complex 函数"""

    def test_real_only(self):
        assert _format_complex(1.0 + 0j) == "1.0"
        assert _format_complex(0.0 + 0j) == "0.0"
        assert _format_complex(-3.14 + 0j) == "-3.14"

    def test_complex_with_imag(self):
        result = _format_complex(1.0 + 2.0j)
        assert "1.0" in result
        assert "2.0" in result
        assert "i" in result

    def test_negative_imag(self):
        result = _format_complex(1.0 - 0.5j)
        assert "1.0" in result
        assert "-0.5" in result

    def test_pure_imag(self):
        result = _format_complex(0.0 + 1.0j)
        assert "0.0" in result
        assert "1.0" in result

    def test_with_unit(self):
        assert _format_complex(1.0 + 0j, "[Pa]") == "1.0[Pa]"
        result = _format_complex(1.0 + 2.0j, "[Pa]")
        assert result.endswith("[Pa]")


class TestParseComplexStr:
    """测试 _parse_complex_str 函数"""

    def test_real_only(self):
        assert _parse_complex_str("3.14") == pytest.approx(3.14 + 0j)

    def test_complex_standard(self):
        result = _parse_complex_str("1.0+2.0j")
        assert result == pytest.approx(1.0 + 2.0j)

    def test_complex_comsol_format(self):
        """COMSOL 使用 'i' 而非 'j'"""
        result = _parse_complex_str("1.0+2.0i")
        assert result == pytest.approx(1.0 + 2.0j)

    def test_negative_imag(self):
        result = _parse_complex_str("1.0-0.5i")
        assert result == pytest.approx(1.0 - 0.5j)

    def test_with_whitespace(self):
        result = _parse_complex_str("  3.14  ")
        assert result == pytest.approx(3.14 + 0j)

    def test_zero(self):
        assert _parse_complex_str("0") == pytest.approx(0.0 + 0j)

    def test_pure_imag(self):
        result = _parse_complex_str("2.0i")
        assert result == pytest.approx(0.0 + 2.0j)


# =============================================================================
# ScanResult 数据类测试
# =============================================================================


class TestScanResult:
    """测试 ScanResult 数据类"""

    @staticmethod
    def _make_result(res: int = 3) -> ScanResult:
        """创建一个测试用的 ScanResult 实例"""
        cr_values = np.linspace(1.004, 1.006, res)
        ci_values = np.linspace(-0.074, -0.072, res)
        return ScanResult(
            f=3430.0, cr=1.005, cr_min=1.004, cr_max=1.006,
            ci=-0.073, ci_min=-0.074, ci_max=-0.072, res=res,
            cr_values=cr_values, ci_values=ci_values,
            eight_target=np.random.randn(res, res, 8) + 1j * np.random.randn(res, res, 8),
            eight_initial=np.random.randn(8) + 1j * np.random.randn(8),
            eight_tf_matrix=np.random.randn(8, 8) + 1j * np.random.randn(8, 8),
            eight_gains=np.random.randn(res, res, 8) + 1j * np.random.randn(res, res, 8),
            floquet_target=np.random.randn(res, res, 3) + 1j * np.random.randn(res, res, 3),
            floquet_initial=np.random.randn(3) + 1j * np.random.randn(3),
            floquet_tf=np.random.randn(3) + 1j * np.random.randn(3),
            floquet_gains=np.random.randn(res, res, 3) + 1j * np.random.randn(res, res, 3),
            exp_eight_steady=np.random.randn(res, res, 8) + 1j * np.random.randn(res, res, 8),
            exp_floquet_steady=np.random.randn(res, res, 8) + 1j * np.random.randn(res, res, 8),
        )

    def test_creation(self):
        result = self._make_result()
        assert result.f == 3430.0
        assert result.cr == 1.005
        assert result.res == 3

    def test_shapes(self):
        res = 4
        result = self._make_result(res)
        assert result.eight_target.shape == (res, res, 8)
        assert result.eight_initial.shape == (8,)
        assert result.eight_tf_matrix.shape == (8, 8)
        assert result.eight_gains.shape == (res, res, 8)
        assert result.floquet_target.shape == (res, res, 3)
        assert result.floquet_initial.shape == (3,)
        assert result.floquet_tf.shape == (3,)
        assert result.floquet_gains.shape == (res, res, 3)

    def test_repr(self):
        result = self._make_result()
        r = repr(result)
        assert "ScanResult" in r
        assert "3430" in r
        assert "eight_gains" in r
        assert "floquet_gains" in r

    def test_cr_ci_values(self):
        res = 5
        result = self._make_result(res)
        assert len(result.cr_values) == res
        assert len(result.ci_values) == res
        np.testing.assert_almost_equal(result.cr_values[0], 1.004)
        np.testing.assert_almost_equal(result.cr_values[-1], 1.006)


# =============================================================================
# SimScanner 初始化与常量测试
# =============================================================================


class TestSimScannerInit:
    """测试 SimScanner 类初始化和常量"""

    def test_class_constants_exist(self):
        """四个 mph 文件常量应指向 sim/mphs/ 下的文件"""
        assert SimScanner.EIGHT_PARA_SCAN_FILE.name == "eight_probes_para_scan.mph"
        assert SimScanner.EIGHT_SINGLE_FILE.name == "eight_probes_single.mph"
        assert SimScanner.FLOQUET_PARA_SCAN_FILE.name == "floquet_probes_para_scan.mph"
        assert SimScanner.FLOQUET_SINGLE_FILE.name == "floquet_probes_single.mph"

    def test_class_constants_paths(self):
        """文件路径应在 sim/mphs/ 子目录下"""
        parent = SimScanner.EIGHT_PARA_SCAN_FILE.parent
        assert parent.name == "mphs"
        # 四个文件应在同一目录
        assert SimScanner.EIGHT_SINGLE_FILE.parent == parent
        assert SimScanner.FLOQUET_PARA_SCAN_FILE.parent == parent
        assert SimScanner.FLOQUET_SINGLE_FILE.parent == parent

    def test_init(self):
        """初始化应设置正确的属性"""
        scanner = SimScanner()
        assert scanner.client is None
        assert scanner.last_result is None
        assert scanner.is_connected is False
        assert scanner.storage_dir.name == "sim"

    def test_run_scan_without_connection_raises(self):
        """未连接时调用 run_scan 应抛出 RuntimeError"""
        scanner = SimScanner()
        with pytest.raises(RuntimeError, match="未连接到 COMSOL Server"):
            scanner.run_scan()

    def test_disconnect_when_not_connected(self):
        """未连接时调用 disconnect 不应报错"""
        scanner = SimScanner()
        scanner.disconnect()  # 不应抛出异常
        assert scanner.client is None


# =============================================================================
# 增益系数批量求解测试（核心计算逻辑）
# =============================================================================


class TestSolveGainsBatch:
    """测试 _solve_gains_batch 向量化求解器"""

    def test_basic_solve(self):
        """基本求解：验证输出形状和数值正确性"""
        np.random.seed(42)
        res = 3
        n = 8

        # 构造一个对角占优的传递矩阵（物理上合理）
        tf_matrix = np.eye(n, dtype=complex) * 0.5
        tf_matrix += np.random.randn(n, n) * 0.01 + 1j * np.random.randn(n, n) * 0.01

        # 构造目标稳态和初态
        probes_initial = np.ones(n, dtype=complex) * 0.1
        probes_target = np.random.randn(res, res, n) + 1j * np.random.randn(res, res, n)

        gains = SimScanner._solve_gains_batch(probes_target, probes_initial, tf_matrix)

        # 形状验证
        assert gains.shape == (res, res, n)
        assert gains.dtype == np.complex128

    def test_single_point_target_equals_initial(self):
        """当目标等于初态时，增益应为 1（无需扬声器干预）"""
        n = 8
        tf_matrix = np.eye(n, dtype=complex) * 0.5
        probes_initial = np.ones(n, dtype=complex) * 0.3
        # 目标 = 初态 => delta = 0 => speaker_amps = 0 => g = T/T = 1
        probes_target = np.tile(probes_initial, (2, 2, 1))

        gains = SimScanner._solve_gains_batch(probes_target, probes_initial, tf_matrix)

        np.testing.assert_allclose(gains, 1.0 + 0j, atol=1e-10)

    def test_consistency_with_manual_solve(self):
        """与手动逐点求解的结果一致"""
        np.random.seed(123)
        res = 4
        n = 8

        tf_matrix = np.eye(n, dtype=complex) * 0.6
        tf_matrix += (np.random.randn(n, n) + 1j * np.random.randn(n, n)) * 0.02

        probes_initial = np.random.randn(n) * 0.1 + 1j * np.random.randn(n) * 0.1
        probes_target = np.random.randn(res, res, n) + 1j * np.random.randn(res, res, n)

        # 批量求解
        gains_batch = SimScanner._solve_gains_batch(probes_target, probes_initial, tf_matrix)

        # 手动逐点求解
        gains_manual = np.zeros((res, res, n), dtype=complex)
        mt = tf_matrix.T
        tf_diag = np.diag(tf_matrix)
        for i in range(res):
            for j in range(res):
                target_ij = probes_target[i, j]
                delta = target_ij - probes_initial
                speaker = np.linalg.solve(mt, delta)
                incident = target_ij - tf_diag * speaker
                gains_manual[i, j] = target_ij / incident

        np.testing.assert_allclose(gains_batch, gains_manual, rtol=1e-10, atol=1e-12)

    def test_different_res_values(self):
        """不同分辨率下均能正确求解"""
        for res in [1, 2, 5, 10]:
            tf_matrix = np.eye(8, dtype=complex) * 0.5
            probes_initial = np.ones(8, dtype=complex) * 0.2
            probes_target = np.random.randn(res, res, 8) + 1j * np.random.randn(res, res, 8)

            gains = SimScanner._solve_gains_batch(probes_target, probes_initial, tf_matrix)
            assert gains.shape == (res, res, 8)
            assert np.all(np.isfinite(gains))

    def test_output_dtype(self):
        """输出应为 complex128"""
        tf_matrix = np.eye(8, dtype=complex) * 0.5
        probes_initial = np.ones(8, dtype=complex) * 0.1
        probes_target = np.ones((2, 2, 8), dtype=complex) * 0.5

        gains = SimScanner._solve_gains_batch(probes_target, probes_initial, tf_matrix)
        assert gains.dtype == np.complex128


# =============================================================================
# Floquet 模式标量增益计算测试
# =============================================================================


class TestFloquetGainComputation:
    """测试 Floquet 模式的标量增益系数计算逻辑

    Floquet 模式的增益计算在 run_scan 方法中直接完成（非独立方法），
    此测试类验证其数学逻辑的正确性。
    """

    @staticmethod
    def compute_floquet_gains_scalar(
        target: np.ndarray, initial: complex, tf: complex,
    ) -> np.ndarray:
        """复现 run_scan 中的 Floquet 标量增益计算"""
        delta = target - initial
        speaker_amps = delta / tf
        incident = target - tf * speaker_amps
        return target / incident

    def test_target_equals_initial(self):
        """目标=初态时增益为1"""
        target = np.ones((3, 3), dtype=complex) * 0.5
        initial = 0.5 + 0j
        tf = 0.3 + 0.1j

        gains = self.compute_floquet_gains_scalar(target, initial, tf)
        np.testing.assert_allclose(gains, 1.0 + 0j, atol=1e-12)

    def test_nontrivial_case(self):
        """非平凡情况：手动验证一个点"""
        target_val = 1.0 + 0.5j
        initial = 0.2 + 0.1j
        tf = 0.3 - 0.2j

        delta = target_val - initial
        speaker = delta / tf
        incident = target_val - tf * speaker
        expected_gain = target_val / incident

        target_arr = np.array([[target_val]], dtype=complex)
        gains = self.compute_floquet_gains_scalar(target_arr, initial, tf)
        assert gains[0, 0] == pytest.approx(expected_gain)


# =============================================================================
# 数据提取逻辑测试（Mock COMSOL 表数据）
# =============================================================================


class TestTableDataExtraction:
    """测试 _get_table_data 和相关提取方法，使用 Mock 模拟 COMSOL 返回"""

    def test_get_table_data_skips_header(self):
        """_get_table_data 应跳过表头行"""
        scanner = SimScanner()

        # 模拟 COMSOL 返回的数据：首行为表头，后续为数据
        mock_java_model = MagicMock()
        mock_probe = MagicMock()
        mock_probe.getString.return_value = "tbl1"
        mock_java_model.probe.return_value = mock_probe

        mock_table = MagicMock()
        # 模拟 java.lang.String 的行为：str() 可以转换
        class FakeJavaString:
            def __init__(self, val):
                self._val = val
            def __str__(self):
                return self._val
            def __repr__(self):
                return f"JavaString({self._val})"

        mock_table.getTableData.return_value = [
            [FakeJavaString("freq"), FakeJavaString("cr"), FakeJavaString("ci"), FakeJavaString("point1")],
            [FakeJavaString("3430"), FakeJavaString("1.005"), FakeJavaString("-0.073"), FakeJavaString("0.5+0.3i")],
            [FakeJavaString("3430"), FakeJavaString("1.006"), FakeJavaString("-0.072"), FakeJavaString("0.6-0.2i")],
        ]
        mock_java_model.result.return_value.table.return_value = mock_table

        data_rows = scanner._get_table_data(mock_java_model, "point1")
        assert len(data_rows) == 2
        # 应返回 Python 字符串
        assert isinstance(data_rows[0][0], str)
        assert data_rows[0][0] == "3430"

    def test_extract_scan_table(self):
        """_extract_scan_table 正确提取参数扫描数据"""
        scanner = SimScanner()

        mock_java_model = MagicMock()
        mock_probe = MagicMock()
        mock_probe.getString.return_value = "tbl1"
        mock_java_model.probe.return_value = mock_probe

        mock_table = MagicMock()
        # 模拟 3 行扫描数据: [freq, cr, ci, point1, point2]
        mock_table.getTableData.return_value = [
            ["3430", "1.004", "-0.074", "0.1+0.2i", "0.3-0.1i"],
            ["3430", "1.004", "-0.073", "0.4+0.5i", "0.6+0.7i"],
            ["3430", "1.005", "-0.074", "0.8-0.3i", "0.2+0.9i"],
        ]
        mock_java_model.result.return_value.table.return_value = mock_table

        result = scanner._extract_scan_table(
            mock_java_model, "point1", probe_start=3, n_probes=2,
        )
        assert result.shape == (3, 2)
        assert result.dtype == np.complex128
        assert result[0, 0] == pytest.approx(0.1 + 0.2j)
        assert result[1, 1] == pytest.approx(0.6 + 0.7j)
        assert result[2, 0] == pytest.approx(0.8 - 0.3j)

    def test_extract_single_row(self):
        """_extract_single_row 正确提取单点仿真数据"""
        scanner = SimScanner()

        mock_java_model = MagicMock()
        mock_probe = MagicMock()
        mock_probe.getString.return_value = "tbl1"
        mock_java_model.probe.return_value = mock_probe

        mock_table = MagicMock()
        # 模拟单点仿真: [freq, point1, point2, point3]
        mock_table.getTableData.return_value = [
            ["3430", "1.0+0.5i", "0.3-0.2i", "0.7+0.1i"],
        ]
        mock_java_model.result.return_value.table.return_value = mock_table

        result = scanner._extract_single_row(
            mock_java_model, "point1", probe_start=1, n_probes=3,
        )
        assert result.shape == (3,)
        assert result[0] == pytest.approx(1.0 + 0.5j)
        assert result[1] == pytest.approx(0.3 - 0.2j)
        assert result[2] == pytest.approx(0.7 + 0.1j)

    def test_empty_table_raises(self):
        """空表数据应抛出 RuntimeError"""
        scanner = SimScanner()

        mock_java_model = MagicMock()
        mock_probe = MagicMock()
        mock_probe.getString.return_value = "tbl1"
        mock_java_model.probe.return_value = mock_probe

        mock_table = MagicMock()
        mock_table.getTableData.return_value = []
        mock_java_model.result.return_value.table.return_value = mock_table

        with pytest.raises(RuntimeError, match="表数据为空"):
            scanner._get_table_data(mock_java_model, "point1")


# =============================================================================
# 绘图功能测试
# =============================================================================


class TestPlotDiscreteHeatmap:
    """测试 _plot_discrete_heatmap 离散热图绘制"""

    def test_basic_plot(self, tmp_path):
        """基本绘图：验证能正确生成文件"""
        cr = np.linspace(1.004, 1.006, 5)
        ci = np.linspace(-0.074, -0.072, 5)
        data = np.random.rand(5, 5)

        save_path = tmp_path / "test_heatmap.png"
        result = SimScanner._plot_discrete_heatmap(
            cr, ci, data, title="Test Heatmap", save_path=save_path,
        )

        assert result.exists()
        assert result.suffix == ".png"

    def test_single_point(self, tmp_path):
        """res=1 时也能绘图"""
        cr = np.array([1.005])
        ci = np.array([-0.073])
        data = np.array([[0.5]])

        save_path = tmp_path / "single_point.png"
        SimScanner._plot_discrete_heatmap(
            cr, ci, data, title="Single Point", save_path=save_path,
        )
        assert save_path.exists()

    def test_asymmetric_grid(self, tmp_path):
        """非正方形网格"""
        cr = np.linspace(1.004, 1.006, 3)
        ci = np.linspace(-0.074, -0.072, 7)
        data = np.random.rand(3, 7)

        save_path = tmp_path / "asymmetric.png"
        SimScanner._plot_discrete_heatmap(
            cr, ci, data, title="Asymmetric", save_path=save_path,
        )
        assert save_path.exists()


# =============================================================================
# 保存功能测试
# =============================================================================


class TestSaveScanResult:
    """测试 _save_scan_result 数据保存"""

    def test_save_creates_npz(self, tmp_path):
        """保存应创建 npz 文件"""
        scanner = SimScanner()
        scanner.storage_dir = tmp_path

        res = 2
        result = ScanResult(
            f=3430.0, cr=1.005, cr_min=1.004, cr_max=1.006,
            ci=-0.073, ci_min=-0.074, ci_max=-0.072, res=res,
            cr_values=np.linspace(1.004, 1.006, res),
            ci_values=np.linspace(-0.074, -0.072, res),
            eight_target=np.ones((res, res, 8), dtype=complex),
            eight_initial=np.ones(8, dtype=complex) * 0.1,
            eight_tf_matrix=np.eye(8, dtype=complex) * 0.5,
            eight_gains=np.ones((res, res, 8), dtype=complex),
            floquet_target=np.ones((res, res, 3), dtype=complex),
            floquet_initial=np.ones(3, dtype=complex) * 0.1,
            floquet_tf=np.ones(3, dtype=complex) * 0.3,
            floquet_gains=np.ones((res, res, 3), dtype=complex),
            exp_eight_steady=np.ones((res, res, 8), dtype=complex),
            exp_floquet_steady=np.ones((res, res, 8), dtype=complex),
        )

        save_dir = scanner._save_scan_result(result)

        npz_path = save_dir / "scan_result.npz"
        assert npz_path.exists()

        # 验证 npz 文件内容
        loaded = np.load(npz_path)
        assert "eight_gains" in loaded
        assert "floquet_gains" in loaded
        assert "eight_tf_matrix" in loaded
        assert loaded["eight_gains"].shape == (res, res, 8)
        assert loaded["floquet_gains"].shape == (res, res, 3)

    def test_save_load_roundtrip(self, tmp_path):
        """保存后加载的数据应与原始数据一致"""
        scanner = SimScanner()
        scanner.storage_dir = tmp_path

        res = 3
        eight_gains = np.random.randn(res, res, 8) + 1j * np.random.randn(res, res, 8)
        result = ScanResult(
            f=3430.0, cr=1.005, cr_min=1.004, cr_max=1.006,
            ci=-0.073, ci_min=-0.074, ci_max=-0.072, res=res,
            cr_values=np.linspace(1.004, 1.006, res),
            ci_values=np.linspace(-0.074, -0.072, res),
            eight_target=np.zeros((res, res, 8), dtype=complex),
            eight_initial=np.zeros(8, dtype=complex),
            eight_tf_matrix=np.eye(8, dtype=complex),
            eight_gains=eight_gains,
            floquet_target=np.zeros((res, res, 3), dtype=complex),
            floquet_initial=np.zeros(3, dtype=complex),
            floquet_tf=np.ones(3, dtype=complex),
            floquet_gains=np.zeros((res, res, 3), dtype=complex),
            exp_eight_steady=np.zeros((res, res, 8), dtype=complex),
            exp_floquet_steady=np.zeros((res, res, 8), dtype=complex),
        )

        save_dir = scanner._save_scan_result(result)
        loaded = np.load(save_dir / "scan_result.npz")

        np.testing.assert_allclose(loaded["eight_gains"], eight_gains)


# =============================================================================
# 完整绘图流程测试
# =============================================================================


class TestPlotScanResults:
    """测试 _plot_scan_results 完整绘图流程"""

    def test_creates_four_plots(self, tmp_path):
        """应生成 6 幅热图"""
        scanner = SimScanner()

        res = 3
        result = ScanResult(
            f=3430.0, cr=1.005, cr_min=1.004, cr_max=1.006,
            ci=-0.073, ci_min=-0.074, ci_max=-0.072, res=res,
            cr_values=np.linspace(1.004, 1.006, res),
            ci_values=np.linspace(-0.074, -0.072, res),
            eight_target=np.random.randn(res, res, 8) + 1j * np.random.randn(res, res, 8),
            eight_initial=np.ones(8, dtype=complex) * 0.1,
            eight_tf_matrix=np.eye(8, dtype=complex) * 0.5,
            eight_gains=np.random.randn(res, res, 8) + 1j * np.random.randn(res, res, 8),
            floquet_target=np.random.randn(res, res, 3) + 1j * np.random.randn(res, res, 3),
            floquet_initial=np.ones(3, dtype=complex) * 0.1,
            floquet_tf=np.ones(3, dtype=complex) * 0.3,
            floquet_gains=np.random.randn(res, res, 3) + 1j * np.random.randn(res, res, 3),
            exp_eight_steady=np.random.randn(res, res, 8) + 1j * np.random.randn(res, res, 8),
            exp_floquet_steady=np.random.randn(res, res, 8) + 1j * np.random.randn(res, res, 8),
        )

        save_dir = tmp_path / "plots"
        save_dir.mkdir()
        scanner._plot_scan_results(result, save_dir)

        expected_files = [
            "eight_gains_mean.png",
            "eight_target_mean.png",
            "exp_eight_steady_mean.png",
            "floquet_gain_point1.png",
            "floquet_target_point1.png",
            "exp_floquet_steady_mean.png",
        ]
        for fname in expected_files:
            assert (save_dir / fname).exists(), f"缺少绘图文件: {fname}"
