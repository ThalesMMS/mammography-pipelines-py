"""
Unit tests for wizard command (menu navigation, option selection, command generation).

Tests cover:
- Menu navigation with _ask_choice
- Input helpers (_ask_yes_no, _ask_string, _ask_int, _ask_float, _ask_optional)
- Dataset path validation
- Command building (_build_cli_command)
- Wizard workflows (train, embed, inference, etc.)
- Run wizard main entry point
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import Mock, patch, call

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mammography import wizard as wizard_module


class TestAskChoice:
    """Test menu navigation with _ask_choice."""

    def test_ask_choice_with_valid_input(self) -> None:
        """Test that _ask_choice returns selected index."""
        with patch("builtins.input", return_value="1"):
            result = wizard_module._ask_choice(
                "Choose an option:", ["Option A", "Option B", "Option C"], default=0
            )
            assert result == 1

    def test_ask_choice_with_default(self) -> None:
        """Test that empty input returns default."""
        with patch("builtins.input", return_value=""):
            result = wizard_module._ask_choice(
                "Choose an option:", ["Option A", "Option B"], default=1
            )
            assert result == 1

    def test_ask_choice_with_zero(self) -> None:
        """Test that zero index is accepted."""
        with patch("builtins.input", return_value="0"):
            result = wizard_module._ask_choice(
                "Choose an option:", ["First", "Second"], default=1
            )
            assert result == 0

    def test_ask_choice_rejects_out_of_range(self) -> None:
        """Test that out-of-range indices are rejected and re-prompted."""
        inputs = ["5", "2"]  # First invalid, second valid
        with patch("builtins.input", side_effect=inputs):
            result = wizard_module._ask_choice(
                "Choose:", ["A", "B", "C"], default=0
            )
            assert result == 2

    def test_ask_choice_rejects_non_numeric(self) -> None:
        """Test that non-numeric input is rejected."""
        inputs = ["invalid", "abc", "1"]  # Invalid inputs, then valid
        with patch("builtins.input", side_effect=inputs):
            result = wizard_module._ask_choice(
                "Choose:", ["X", "Y", "Z"], default=0
            )
            assert result == 1


class TestAskYesNo:
    """Test yes/no prompts."""

    def test_ask_yes_no_yes_responses(self) -> None:
        """Test that various 'yes' responses return True."""
        yes_inputs = ["y", "yes", "Y", "YES", "s", "sim"]
        for user_input in yes_inputs:
            with patch("builtins.input", return_value=user_input):
                result = wizard_module._ask_yes_no("Proceed?", default=False)
                assert result is True

    def test_ask_yes_no_no_responses(self) -> None:
        """Test that various 'no' responses return False."""
        no_inputs = ["n", "no", "N", "NO", "nao"]
        for user_input in no_inputs:
            with patch("builtins.input", return_value=user_input):
                result = wizard_module._ask_yes_no("Proceed?", default=True)
                assert result is False

    def test_ask_yes_no_default_true(self) -> None:
        """Test that empty input returns default (True)."""
        with patch("builtins.input", return_value=""):
            result = wizard_module._ask_yes_no("Proceed?", default=True)
            assert result is True

    def test_ask_yes_no_default_false(self) -> None:
        """Test that empty input returns default (False)."""
        with patch("builtins.input", return_value=""):
            result = wizard_module._ask_yes_no("Proceed?", default=False)
            assert result is False

    def test_ask_yes_no_retries_on_invalid(self) -> None:
        """Test that invalid input prompts retry."""
        inputs = ["maybe", "perhaps", "y"]
        with patch("builtins.input", side_effect=inputs):
            result = wizard_module._ask_yes_no("Proceed?", default=False)
            assert result is True


class TestAskString:
    """Test string input helper."""

    def test_ask_string_with_input(self) -> None:
        """Test that user input is returned."""
        with patch("builtins.input", return_value="custom_value"):
            result = wizard_module._ask_string("Enter value:", default="default")
            assert result == "custom_value"

    def test_ask_string_with_empty_uses_default(self) -> None:
        """Test that empty input uses default."""
        with patch("builtins.input", return_value=""):
            result = wizard_module._ask_string("Enter value:", default="default")
            assert result == "default"

    def test_ask_string_without_default(self) -> None:
        """Test string input without default."""
        with patch("builtins.input", return_value="value"):
            result = wizard_module._ask_string("Enter value:")
            assert result == "value"

    def test_ask_string_strips_whitespace(self) -> None:
        """Test that input is stripped."""
        with patch("builtins.input", return_value="  trimmed  "):
            result = wizard_module._ask_string("Enter value:")
            assert result == "trimmed"


class TestAskInt:
    """Test integer input helper."""

    def test_ask_int_with_valid_input(self) -> None:
        """Test that valid integer is returned."""
        with patch("builtins.input", return_value="42"):
            result = wizard_module._ask_int("Enter number:", default=10)
            assert result == 42

    def test_ask_int_with_empty_uses_default(self) -> None:
        """Test that empty input uses default."""
        with patch("builtins.input", return_value=""):
            result = wizard_module._ask_int("Enter number:", default=10)
            assert result == 10

    def test_ask_int_retries_on_invalid(self) -> None:
        """Test that invalid input prompts retry."""
        inputs = ["not_a_number", "3.14", "100"]
        with patch("builtins.input", side_effect=inputs):
            result = wizard_module._ask_int("Enter number:", default=10)
            assert result == 100

    def test_ask_int_accepts_zero(self) -> None:
        """Test that zero is accepted."""
        with patch("builtins.input", return_value="0"):
            result = wizard_module._ask_int("Enter number:", default=10)
            assert result == 0


class TestAskFloat:
    """Test float input helper."""

    def test_ask_float_with_valid_input(self) -> None:
        """Test that valid float is returned."""
        with patch("builtins.input", return_value="3.14"):
            result = wizard_module._ask_float("Enter value:", default=1.0)
            assert result == 3.14

    def test_ask_float_with_empty_uses_default(self) -> None:
        """Test that empty input uses default."""
        with patch("builtins.input", return_value=""):
            result = wizard_module._ask_float("Enter value:", default=2.5)
            assert result == 2.5

    def test_ask_float_accepts_integer(self) -> None:
        """Test that integer input is converted to float."""
        with patch("builtins.input", return_value="42"):
            result = wizard_module._ask_float("Enter value:", default=1.0)
            assert result == 42.0

    def test_ask_float_retries_on_invalid(self) -> None:
        """Test that invalid input prompts retry."""
        inputs = ["not_a_number", "abc", "1.5"]
        with patch("builtins.input", side_effect=inputs):
            result = wizard_module._ask_float("Enter value:", default=1.0)
            assert result == 1.5


class TestAskOptional:
    """Test optional input helper."""

    def test_ask_optional_with_input(self) -> None:
        """Test that input is returned."""
        with patch("builtins.input", return_value="value"):
            result = wizard_module._ask_optional("Enter optional:")
            assert result == "value"

    def test_ask_optional_with_empty(self) -> None:
        """Test that empty input returns None."""
        with patch("builtins.input", return_value=""):
            result = wizard_module._ask_optional("Enter optional:")
            assert result is None

    def test_ask_optional_strips_whitespace(self) -> None:
        """Test that input is stripped."""
        with patch("builtins.input", return_value="  value  "):
            result = wizard_module._ask_optional("Enter optional:")
            assert result == "value"


class TestBuildCliCommand:
    """Test CLI command building."""

    def test_build_cli_command_basic(self) -> None:
        """Test basic command construction."""
        cmd = wizard_module._build_cli_command("train-density", ["--epochs", "10"])
        assert cmd[0] == sys.executable
        assert cmd[1:4] == ["-m", "mammography.cli", "train-density"]
        assert "--epochs" in cmd
        assert "10" in cmd

    def test_build_cli_command_with_multiple_args(self) -> None:
        """Test command with multiple arguments."""
        args = ["--arch", "resnet50", "--batch-size", "16", "--epochs", "5"]
        cmd = wizard_module._build_cli_command("embed", args)
        assert cmd[0] == sys.executable
        assert "embed" in cmd
        assert "--arch" in cmd
        assert "resnet50" in cmd
        assert "--batch-size" in cmd
        assert "16" in cmd

    def test_build_cli_command_preserves_order(self) -> None:
        """Test that argument order is preserved."""
        args = ["--a", "1", "--b", "2", "--c", "3"]
        cmd = wizard_module._build_cli_command("test", args)
        # Extract just the args portion
        args_portion = cmd[4:]
        assert args_portion == args


class TestValidateDatasetPath:
    """Test dataset path validation."""

    def test_validate_empty_path(self) -> None:
        """Test that empty path returns False."""
        result = wizard_module._validate_dataset_path("", None)
        assert result is False

    def test_validate_nonexistent_path(self) -> None:
        """Test that nonexistent path returns False."""
        result = wizard_module._validate_dataset_path("/nonexistent/path", None)
        assert result is False

    def test_validate_existing_csv_file(self, tmp_path: Path) -> None:
        """Test that existing CSV file is validated."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("col1,col2\nval1,val2", encoding="utf-8")
        result = wizard_module._validate_dataset_path(str(csv_file), None)
        assert result is True

    def test_validate_csv_path_not_a_file(self, tmp_path: Path) -> None:
        """Test that directory with .csv extension fails."""
        csv_dir = tmp_path / "test.csv"
        csv_dir.mkdir()
        result = wizard_module._validate_dataset_path(str(csv_dir), None)
        assert result is False

    def test_validate_directory_with_format_detection(self, tmp_path: Path) -> None:
        """Test directory validation with format detection."""
        # Create a directory with some images
        (tmp_path / "image1.png").touch()
        (tmp_path / "image2.png").touch()

        with patch("mammography.wizard.detect_dataset_format") as mock_detect:
            # Mock format detection
            mock_format = Mock()
            mock_format.dataset_type = "custom"
            mock_format.image_format = "png"
            mock_format.image_count = 2
            mock_format.csv_path = None
            mock_format.dicom_root = None
            mock_format.format_counts = {"png": 2}
            mock_detect.return_value = mock_format

            with patch("mammography.wizard.validate_format", return_value=[]):
                result = wizard_module._validate_dataset_path(str(tmp_path), "custom")
                assert result is True

    def test_validate_directory_with_no_images(self, tmp_path: Path) -> None:
        """Test that directory with no images returns False."""
        with patch("mammography.wizard.detect_dataset_format") as mock_detect:
            mock_format = Mock()
            mock_format.dataset_type = "custom"
            mock_format.image_format = "unknown"
            mock_format.image_count = 0
            mock_format.csv_path = None
            mock_format.dicom_root = None
            mock_format.format_counts = {}  # Empty dict instead of Mock
            mock_format.warnings = []  # Empty list for warnings
            mock_format.suggestions = []  # Empty list for suggestions
            mock_detect.return_value = mock_format

            result = wizard_module._validate_dataset_path(str(tmp_path), None)
            assert result is False


class TestWizardCommand:
    """Test WizardCommand dataclass."""

    def test_wizard_command_creation(self) -> None:
        """Test creating WizardCommand."""
        cmd = wizard_module.WizardCommand(
            label="Test Command",
            argv=["python", "-m", "test"]
        )
        assert cmd.label == "Test Command"
        assert cmd.argv == ["python", "-m", "test"]


class TestWizardQuickTrain:
    """Test quick train wizard workflow."""

    def test_wizard_quick_train_generates_command(self) -> None:
        """Test that quick train generates valid command."""
        # Mock all user inputs
        inputs = [
            "",  # config path (empty)
            "0",  # dataset preset (archive)
            "classificacao.csv",  # CSV path
            "archive",  # DICOM root
            "0",  # architecture choice (efficientnet_b0)
            "outputs/quick_train",  # outdir
        ]

        with patch("builtins.input", side_effect=inputs):
            with patch("mammography.wizard._validate_dataset_path", return_value=True):
                cmd = wizard_module._wizard_quick_train()

                assert cmd.label == "Treino rapido"
                assert "--arch" in cmd.argv
                assert "efficientnet_b0" in cmd.argv
                assert "--classes" in cmd.argv
                assert "binary" in cmd.argv
                assert "--epochs" in cmd.argv
                assert "5" in cmd.argv
                assert "--batch-size" in cmd.argv
                assert "16" in cmd.argv


class TestWizardInference:
    """Test inference wizard workflow."""

    def test_wizard_inference_generates_command(self) -> None:
        """Test that inference wizard generates valid command."""
        inputs = [
            "",  # config (empty)
            "model.pt",  # checkpoint
            "input_dir",  # input path
            "0",  # arch (resnet50)
            "0",  # classes (multiclass)
            "224",  # img_size
            "16",  # batch_size
            "auto",  # device
            "",  # output csv (empty)
            "n",  # custom normalization
            "n",  # use AMP
            "",  # extra args (empty)
        ]

        with patch("builtins.input", side_effect=inputs):
            cmd = wizard_module._wizard_inference()

            assert cmd.label == "Inferencia"
            assert "--checkpoint" in cmd.argv
            assert "model.pt" in cmd.argv
            assert "--input" in cmd.argv
            assert "input_dir" in cmd.argv
            assert "--arch" in cmd.argv
            assert "resnet50" in cmd.argv
            assert "--classes" in cmd.argv
            assert "multiclass" in cmd.argv


class TestWizardAugment:
    """Test augment wizard workflow."""

    def test_wizard_augment_generates_command(self) -> None:
        """Test that augment wizard generates valid command."""
        inputs = [
            "",  # config (empty)
            "source_dir",  # source directory
            "output_dir",  # output directory
            "3",  # num augmentations
            "",  # extra args (empty)
        ]

        with patch("builtins.input", side_effect=inputs):
            cmd = wizard_module._wizard_augment()

            assert cmd.label == "Augmentacao de dados"
            assert "--source-dir" in cmd.argv
            assert "source_dir" in cmd.argv
            assert "--output-dir" in cmd.argv
            assert "output_dir" in cmd.argv
            assert "--num-augmentations" in cmd.argv
            assert "3" in cmd.argv


class TestWizardLabelCommands:
    """Test label wizard workflows."""

    def test_wizard_label_density(self) -> None:
        """Test label-density command generation."""
        cmd = wizard_module._wizard_label_density()
        assert cmd.label == "Rotulagem de densidade"
        assert "label-density" in cmd.argv

    def test_wizard_label_patches(self) -> None:
        """Test label-patches command generation."""
        cmd = wizard_module._wizard_label_patches()
        assert cmd.label == "Rotulagem de patches"
        assert "label-patches" in cmd.argv


class TestRunWizard:
    """Test main wizard entry point."""

    def test_run_wizard_exits_on_last_option(self) -> None:
        """Test that selecting last option (Sair) exits."""
        with patch("builtins.input", return_value="14"):  # Last option index
            result = wizard_module.run_wizard(dry_run=True)
            assert result == 0

    def test_run_wizard_train_option(self) -> None:
        """Test selecting train option."""
        # Mock inputs: 0 (train), then minimal train inputs, then cancel execution
        inputs = [
            "0",  # Select train
            "",  # config (empty)
            "0",  # dataset preset (archive)
            "classificacao.csv",  # CSV
            "archive",  # DICOM root
        ]

        # Mock _wizard_train to avoid full workflow
        with patch("builtins.input", side_effect=inputs):
            with patch("mammography.wizard._validate_dataset_path", return_value=True):
                with patch("mammography.wizard._wizard_train") as mock_train:
                    mock_cmd = wizard_module.WizardCommand(
                        label="Test",
                        argv=["python", "-m", "test"]
                    )
                    mock_train.return_value = mock_cmd

                    with patch("mammography.wizard._run_command", return_value=0):
                        result = wizard_module.run_wizard(dry_run=True)
                        mock_train.assert_called_once()
                        assert result == 0

    def test_run_wizard_embed_option(self) -> None:
        """Test selecting embed option."""
        with patch("builtins.input", return_value="2"):  # Embed option
            with patch("mammography.wizard._wizard_embed") as mock_embed:
                mock_cmd = wizard_module.WizardCommand(
                    label="Test",
                    argv=["python", "-m", "test"]
                )
                mock_embed.return_value = mock_cmd

                with patch("mammography.wizard._run_command", return_value=0):
                    result = wizard_module.run_wizard(dry_run=True)
                    mock_embed.assert_called_once()
                    assert result == 0

    def test_run_wizard_inference_option(self) -> None:
        """Test selecting inference option."""
        with patch("builtins.input", return_value="5"):  # Inference option
            with patch("mammography.wizard._wizard_inference") as mock_inference:
                mock_cmd = wizard_module.WizardCommand(
                    label="Test",
                    argv=["python", "-m", "test"]
                )
                mock_inference.return_value = mock_cmd

                with patch("mammography.wizard._run_command", return_value=0):
                    result = wizard_module.run_wizard(dry_run=True)
                    mock_inference.assert_called_once()
                    assert result == 0


class TestRunCommand:
    """Test command execution."""

    def test_run_command_dry_run(self) -> None:
        """Test that dry-run mode doesn't execute command."""
        cmd = wizard_module.WizardCommand(
            label="Test",
            argv=["echo", "test"]
        )

        with patch("builtins.input", return_value="y"):
            with patch("subprocess.run") as mock_run:
                result = wizard_module._run_command(cmd, dry_run=True)
                assert result == 0
                mock_run.assert_not_called()

    def test_run_command_user_cancels(self) -> None:
        """Test that user can cancel command execution."""
        cmd = wizard_module.WizardCommand(
            label="Test",
            argv=["echo", "test"]
        )

        with patch("builtins.input", return_value="n"):
            with patch("subprocess.run") as mock_run:
                result = wizard_module._run_command(cmd, dry_run=False)
                assert result == 0
                mock_run.assert_not_called()

    def test_run_command_executes(self) -> None:
        """Test that command is executed when confirmed."""
        cmd = wizard_module.WizardCommand(
            label="Test",
            argv=["echo", "test"]
        )

        with patch("builtins.input", return_value="y"):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = Mock(returncode=0)
                result = wizard_module._run_command(cmd, dry_run=False)
                assert result == 0
                mock_run.assert_called_once()


class TestDatasetPrompt:
    """Test dataset prompt workflow."""

    def test_dataset_prompt_archive_preset(self) -> None:
        """Test archive preset selection."""
        inputs = [
            "0",  # Select archive
            "classificacao.csv",
            "archive",
        ]

        with patch("builtins.input", side_effect=inputs):
            with patch("mammography.wizard._validate_dataset_path", return_value=True):
                args, csv_path, dicom_root = wizard_module._dataset_prompt()

                assert "--dataset" in args
                assert "archive" in args
                assert "--csv" in args
                assert "classificacao.csv" in args
                assert "--dicom-root" in args
                assert "archive" in args
                assert csv_path == "classificacao.csv"
                assert dicom_root == "archive"

    def test_dataset_prompt_mamografias_preset(self) -> None:
        """Test mamografias preset selection."""
        inputs = [
            "1",  # Select mamografias
            "mamografias",
        ]

        with patch("builtins.input", side_effect=inputs):
            with patch("mammography.wizard._validate_dataset_path", return_value=True):
                args, csv_path, dicom_root = wizard_module._dataset_prompt()

                assert "--dataset" in args
                assert "mamografias" in args
                assert "--csv" in args
                assert "mamografias" in args
                assert csv_path == "mamografias"
                assert dicom_root is None

    def test_dataset_prompt_custom(self) -> None:
        """Test custom dataset selection."""
        inputs = [
            "3",  # Select custom (last preset option)
            "custom.csv",
            "",  # No DICOM root
        ]

        with patch("builtins.input", side_effect=inputs):
            with patch("mammography.wizard._validate_dataset_path", return_value=True):
                args, csv_path, dicom_root = wizard_module._dataset_prompt()

                assert "--dataset" not in args
                assert "--csv" in args
                assert "custom.csv" in args
                assert csv_path == "custom.csv"
                assert dicom_root is None


class TestPrintProgress:
    """Test progress indicator."""

    def test_print_progress_basic(self, capsys) -> None:
        """Test that progress is printed."""
        wizard_module._print_progress(1, 3, "Test Section")
        captured = capsys.readouterr()
        assert "Passo 1 de 3" in captured.out
        assert "Test Section" in captured.out

    def test_print_progress_without_section(self, capsys) -> None:
        """Test progress without section name."""
        wizard_module._print_progress(2, 5)
        captured = capsys.readouterr()
        assert "Passo 2 de 5" in captured.out
