"""Tests for parameter precedence in extract()."""

from unittest import mock

from absl.testing import absltest

import langcore as lx
from langcore import factory
from langcore.core import data


class ExtractParameterPrecedenceTest(absltest.TestCase):
    """Tests ensuring correct precedence among extract() parameters."""

    def setUp(self):
        super().setUp()
        self.examples = [
            data.ExampleData(
                text="example",
                extractions=[
                    data.Extraction(
                        extraction_class="entity",
                        extraction_text="example",
                    )
                ],
            )
        ]
        self.description = "description"

    @mock.patch("langcore.annotation.Annotator")
    @mock.patch("langcore.extraction.factory.create_model")
    def test_model_overrides_all_other_parameters(
        self, mock_create_model, mock_annotator_cls
    ):
        """Test that model parameter overrides all other model-related parameters."""
        provided_model = mock.MagicMock()
        mock_annotator = mock_annotator_cls.return_value
        mock_annotator.annotate_text.return_value = "ok"

        config = factory.ModelConfig(model_id="config-id")

        result = lx.extract(
            text_or_documents="text",
            prompt_description=self.description,
            examples=self.examples,
            model=provided_model,
            config=config,
            model_id="ignored-model",
            api_key="ignored-key",
            use_schema_constraints=False,
        )

        mock_create_model.assert_not_called()
        _, kwargs = mock_annotator_cls.call_args
        self.assertIs(kwargs["language_model"], provided_model)
        self.assertEqual(result, "ok")

    @mock.patch("langcore.annotation.Annotator")
    @mock.patch("langcore.extraction.factory.create_model")
    def test_config_overrides_model_id(self, mock_create_model, mock_annotator_cls):
        """Test that config parameter overrides model_id."""
        config = factory.ModelConfig(
            model_id="config-model", provider_kwargs={"api_key": "config-key"}
        )
        mock_model = mock.MagicMock()
        mock_model.requires_fence_output = True
        mock_create_model.return_value = mock_model
        mock_annotator = mock_annotator_cls.return_value
        mock_annotator.annotate_text.return_value = "ok"

        with mock.patch("langcore.extraction.factory.ModelConfig") as mock_model_config:
            result = lx.extract(
                text_or_documents="text",
                prompt_description=self.description,
                examples=self.examples,
                config=config,
                model_id="other-model",
                api_key="other-key",
                use_schema_constraints=False,
            )
            mock_model_config.assert_not_called()

        mock_create_model.assert_called_once()
        called_config = mock_create_model.call_args[1]["config"]
        self.assertEqual(called_config.model_id, "config-model")
        self.assertEqual(called_config.provider_kwargs, {"api_key": "config-key"})

        _, kwargs = mock_annotator_cls.call_args
        self.assertIs(kwargs["language_model"], mock_model)
        self.assertEqual(result, "ok")

    @mock.patch("langcore.annotation.Annotator")
    @mock.patch("langcore.extraction.factory.create_model")
    def test_model_id_and_base_kwargs(self, mock_create_model, mock_annotator_cls):
        """Test that model_id and other kwargs are used when no model or config."""
        mock_model = mock.MagicMock()
        mock_model.requires_fence_output = True
        mock_create_model.return_value = mock_model
        mock_annotator_cls.return_value.annotate_text.return_value = "ok"
        mock_config = mock.MagicMock()

        with mock.patch(
            "langcore.extraction.factory.ModelConfig", return_value=mock_config
        ) as mock_model_config:
            result = lx.extract(
                text_or_documents="text",
                prompt_description=self.description,
                examples=self.examples,
                model_id="model-123",
                api_key="api-key",
                temperature=0.9,
                model_url="http://model",
                use_schema_constraints=False,
            )

        mock_model_config.assert_called_once()
        _, kwargs = mock_model_config.call_args
        self.assertEqual(kwargs["model_id"], "model-123")
        provider_kwargs = kwargs["provider_kwargs"]
        self.assertEqual(provider_kwargs["api_key"], "api-key")
        self.assertEqual(provider_kwargs["temperature"], 0.9)
        self.assertEqual(provider_kwargs["model_url"], "http://model")
        self.assertEqual(provider_kwargs["base_url"], "http://model")
        mock_create_model.assert_called_once()
        self.assertEqual(result, "ok")

    @mock.patch("langcore.annotation.Annotator")
    @mock.patch("langcore.extraction.factory.create_model")
    def test_use_schema_constraints_warns_with_config(
        self, mock_create_model, mock_annotator_cls
    ):
        """Test that use_schema_constraints emits warning when used with config."""
        config = factory.ModelConfig(
            model_id="gemini-2.5-flash", provider_kwargs={"api_key": "test-key"}
        )

        mock_model = mock.MagicMock()
        mock_model.requires_fence_output = True
        mock_create_model.return_value = mock_model
        mock_annotator = mock_annotator_cls.return_value
        mock_annotator.annotate_text.return_value = "ok"

        with self.assertWarns(UserWarning) as cm:
            result = lx.extract(
                text_or_documents="text",
                prompt_description=self.description,
                examples=self.examples,
                config=config,
                use_schema_constraints=True,
            )

        self.assertIn("schema constraints", str(cm.warning))
        self.assertIn("applied", str(cm.warning))
        mock_create_model.assert_called_once()
        called_config = mock_create_model.call_args[1]["config"]
        self.assertEqual(called_config.model_id, "gemini-2.5-flash")
        self.assertEqual(result, "ok")

    @mock.patch("langcore.annotation.Annotator")
    @mock.patch("langcore.extraction.factory.create_model")
    def test_use_schema_constraints_warns_with_model(
        self, mock_create_model, mock_annotator_cls
    ):
        """Test that use_schema_constraints emits warning when used with model."""
        provided_model = mock.MagicMock()
        mock_annotator = mock_annotator_cls.return_value
        mock_annotator.annotate_text.return_value = "ok"

        with self.assertWarns(UserWarning) as cm:
            result = lx.extract(
                text_or_documents="text",
                prompt_description=self.description,
                examples=self.examples,
                model=provided_model,
                use_schema_constraints=True,
            )

        self.assertIn("use_schema_constraints", str(cm.warning))
        self.assertIn("ignored", str(cm.warning))
        mock_create_model.assert_not_called()
        self.assertEqual(result, "ok")


if __name__ == "__main__":
    absltest.main()
