import tempfile
import unittest
from pathlib import Path

import eigcm_scheduler as es


class SchedulerTests(unittest.TestCase):
    def make_case(self, tensor=True):
        temp = tempfile.TemporaryDirectory()
        root = Path(temp.name)
        representation = "TensorTree4" if tensor else "Columns"
        (root / "device.xml").write_text(
            """<?xml version="1.0"?>
<WindowElement><Optical><Layer><DataDefinition>
<IncidentDataStructure>{}</IncidentDataStructure>
</DataDefinition><WavelengthData><WavelengthDataBlock>
<AngleBasis>LBNL/Shirley-Chiu</AngleBasis>
</WavelengthDataBlock></WavelengthData></Layer></Optical></WindowElement>
""".format(representation),
            encoding="utf-8",
        )
        (root / "materials.rad").write_text(
            """void BSDF prism
6 0 device.xml 0 0 1 .
0
0
void metal reflector
0
0
5 0.7 0.7 0.7 0.25 0.01
void trans diffuser
0
0
7 0.6 0.6 0.6 0.05 0.08 0.7 0.2
""",
            encoding="utf-8",
        )
        (root / "black.rad").write_text(
            "void plastic black\n0\n0\n5 0 0 0 0 0\n", encoding="utf-8"
        )
        (root / "scene.rad").write_text(
            """prism polygon window
0
0
12 0 0 0 1 0 0 1 1 0 0 1 0
reflector polygon mirror_surface
0
0
12 0 0 0 1 0 0 1 0 1 0 0 1
diffuser polygon diffusing_window
0
0
12 0 0 0 0 1 0 0 1 1 0 0 1
""",
            encoding="utf-8",
        )
        (root / "open.rad").write_text((root / "scene.rad").read_text(), encoding="utf-8")
        (root / "views.txt").write_text("0 0 1 0 1 0\n1 0 1 0 1 0\n", encoding="utf-8")
        for name in ("weather.wea", "sky.rad", "receiver.rad"):
            (root / name).write_text("placeholder\n", encoding="utf-8")
        config = root / "case.cfg"
        config.write_text(
            """[scene]
out = {root}/out
views = {root}/views.txt
nview = auto
nproc = 4
weather = {root}/weather.wea
sky_receiver = {root}/sky.rad
materials = {root}/materials.rad
black_materials = {root}/black.rad
scene = {root}/scene.rad

[workflow]
quality = balanced
directlobecontrast = auto
specularcontrast = auto
bsdf_matrix = auto
ttsuncontrast = auto
ev_strategy = auto

[radiance]
mf = 1

[bsdf]

[windowgroup:main]
receiver = {root}/receiver.rad
mode = bsdf
xml = {root}/device.xml
open_scene = {root}/open.rad
open_materials = {root}/materials.rad
open_black_materials = {root}/black.rad
""".format(root=root.as_posix()),
            encoding="utf-8",
        )
        return temp, config

    def test_tensor_tree_selects_all_specialized_branches(self):
        temp, config = self.make_case(tensor=True)
        self.addCleanup(temp.cleanup)
        settings = es.Settings(str(config))
        detection = es.inspect_scene(settings)
        groups = es.read_window_groups(settings, detection)
        decision = es.choose_workflow(settings, detection, groups)
        self.assertTrue(detection.has_tensortree)
        self.assertTrue(decision.directlobecontrast)
        self.assertTrue(decision.specularcontrast)
        self.assertTrue(decision.bsdf_matrix)
        self.assertTrue(decision.ttsuncontrast)
        self.assertEqual(decision.ev_strategy, "bsdf_sun")
        self.assertEqual(detection.normal_reference, "nearest viewpoint")
        self.assertEqual(detection.normal_reference_count, 2)

    def test_viewpoint_parser_accepts_numeric_and_radiance_view_lines(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        views = Path(temp.name) / "views.txt"
        views.write_text(
            "# comment\n0 1 2 0 1 0\nrvu -vp 3 4 5 -vd 0 1 0\ninvalid\n",
            encoding="utf-8",
        )
        positions, warnings = es.read_view_positions(str(views))
        self.assertEqual(positions, [(0.0, 1.0, 2.0), (3.0, 4.0, 5.0)])
        self.assertEqual(len(warnings), 1)

    def test_nearest_viewpoint_orients_disconnected_room_windows(self):
        material = es.Primitive("void", "glass", "glazing", [], [], [0.9, 0.9, 0.9], "")
        first = es.Primitive(
            "glazing",
            "polygon",
            "window_a",
            [],
            [],
            [0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1],
            "",
        )
        second = es.Primitive(
            "glazing",
            "polygon",
            "window_b",
            [],
            [],
            [10, 0, 0, 10, 1, 0, 10, 1, 1, 10, 0, 1],
            "",
        )
        surfaces = es.detect_fenestration_surfaces(
            [material, first, second],
            False,
            view_positions=[(1, 0.5, 0.5), (11, 0.5, 0.5)],
        )
        self.assertEqual([surface.normal for surface in surfaces], [(1.0, 0.0, 0.0)] * 2)

    def test_explicit_interior_point_overrides_viewpoints(self):
        material = es.Primitive("void", "glass", "glazing", [], [], [0.9, 0.9, 0.9], "")
        window = es.Primitive(
            "glazing",
            "polygon",
            "window",
            [],
            [],
            [0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1],
            "",
        )
        surfaces = es.detect_fenestration_surfaces(
            [material, window],
            False,
            interior_point=(-1, 0.5, 0.5),
            view_positions=[(1, 0.5, 0.5)],
        )
        self.assertEqual(surfaces[0].normal, (-1.0, 0.0, 0.0))

    def test_plan_contains_ordered_outputs(self):
        temp, config = self.make_case(tensor=True)
        self.addCleanup(temp.cleanup)
        plan = es.build_plan(es.Settings(str(config)))
        ids = [step.step_id for step in plan.steps]
        self.assertEqual(plan.nview, 2)
        self.assertIn("directlobecontrast", ids)
        self.assertIn("ttsuncontrast", ids)
        self.assertIn("specularcontrast", ids)
        self.assertEqual(sum(step_id.startswith("bsdf_resample_") for step_id in ids), 1)
        self.assertLess(ids.index("weather"), ids.index("dcglare2"))
        self.assertTrue(plan.steps[-1].outputs[0].endswith("dgp_annual.mtx"))
        self.assertEqual(Path(plan.execution_dir), Path(temp.name))
        self.assertIn("cd {}".format(es.q(plan.execution_dir)), es.render_shell_script(plan))

    def test_forced_disable_is_respected(self):
        temp, config = self.make_case(tensor=True)
        self.addCleanup(temp.cleanup)
        settings = es.Settings(
            str(config),
            [
                "workflow.directlobecontrast=false",
                "workflow.specularcontrast=false",
                "workflow.ttsuncontrast=false",
            ],
        )
        detection = es.inspect_scene(settings)
        decision = es.choose_workflow(
            settings, detection, es.read_window_groups(settings, detection)
        )
        self.assertFalse(decision.directlobecontrast)
        self.assertFalse(decision.specularcontrast)
        self.assertFalse(decision.ttsuncontrast)

    def test_background_only_does_not_subtract_direct_terms(self):
        temp, config = self.make_case(tensor=True)
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        for name in ("tds_total.mtx", "tds_direct.mtx"):
            (root / name).write_text("placeholder\n", encoding="utf-8")
        settings = es.Settings(
            str(config),
            [
                "workflow.directlobecontrast=false",
                "workflow.specularcontrast=false",
                "workflow.bsdf_matrix=false",
                "workflow.ttsuncontrast=false",
                "windowgroup:main.mode=precomputed",
                "windowgroup:main.tds_total={}".format((root / "tds_total.mtx").as_posix()),
                "windowgroup:main.tds_direct={}".format((root / "tds_direct.mtx").as_posix()),
            ],
        )
        plan = es.build_plan(settings)
        background = next(step for step in plan.steps if step.step_id == "ev_background")
        evaluation = next(step for step in plan.steps if step.step_id == "dcglare2")
        self.assertNotIn("+ -s -1", background.command.splitlines()[-1])
        self.assertNotIn("-wVDirect", evaluation.command)

    def test_automatic_window_grouping_and_modes(self):
        temp, config = self.make_case(tensor=True)
        self.addCleanup(temp.cleanup)
        settings = es.Settings(
            str(config),
            [
                "auto_window_groups.enabled=true",
                "auto_window_groups.receiver_offset=0.02",
                "auto_window_groups.open_scene={}".format(
                    (Path(temp.name) / "open.rad").as_posix()
                ),
            ],
        )
        detection = es.inspect_scene(settings)
        groups = es.read_window_groups(settings, detection)
        automatic = [group for group in groups if group.auto_detected]
        self.assertGreaterEqual(len(automatic), 2)
        self.assertIn("bsdf", {group.mode for group in automatic})
        self.assertIn("geometry", {group.mode for group in automatic})
        self.assertTrue(all(group.receiver_content for group in automatic))
        self.assertTrue(
            any("#@rfluxmtx h=r1" in group.receiver_content for group in automatic)
        )


if __name__ == "__main__":
    unittest.main()
