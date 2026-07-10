import unittest

from app import resolve_openai_api_key, should_display_sources


class ResolveOpenAIKeyTests(unittest.TestCase):
    def test_prefers_sidebar_key_when_present(self) -> None:
        self.assertEqual(resolve_openai_api_key("sidebar-key", "env-key"), "sidebar-key")

    def test_falls_back_to_environment_key(self) -> None:
        self.assertEqual(resolve_openai_api_key("", "env-key"), "env-key")

    def test_returns_empty_string_when_no_key_is_available(self) -> None:
        self.assertEqual(resolve_openai_api_key("", ""), "")

    def test_hides_sources_when_answer_says_information_was_not_found(self) -> None:
        self.assertFalse(should_display_sources("I could not find this information in the provided reports.", [object()]))

    def test_shows_sources_for_normal_answers(self) -> None:
        self.assertTrue(should_display_sources("The revenue increased in Q4.", [object()]))


if __name__ == "__main__":
    unittest.main()
