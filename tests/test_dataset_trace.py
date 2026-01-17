import tempfile
import traceback
from pathlib import Path
from typing import Callable
from inference_perf.apis import LazyLoadInferenceAPIData, CompletionAPIData, ChatCompletionAPIData
from inference_perf.datagen.base import LazyLoadDataMixin
from inference_perf.datagen.dataset_trace_datagen import DatasetTraceDataGenerator
from inference_perf.utils.trace_reader import DatasetTraceReader
from inference_perf.config import APIConfig, DataConfig, APIType, TraceFormat, TraceConfig, DataGenType


def test_dataset_trace_reader_basic() -> None:
    """Test basic JSONL parsing with text_input and output_length."""
    content = """{"text_input": "What is the capital of France?", "output_length": 20}
{"text_input": "Explain quantum computing in simple terms.", "output_length": 100}
{"text_input": "Write a Python function for fibonacci.", "output_length": 150}
"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        f.write(content)
        f.flush()
        temp_path = Path(f.name)

    try:
        reader = DatasetTraceReader()
        entries = reader.load_entries(temp_path)

        assert len(entries) == 3, f"Expected 3 entries, got {len(entries)}"
        assert entries[0].text_input == "What is the capital of France?"
        assert entries[0].output_length == 20
        assert entries[1].text_input == "Explain quantum computing in simple terms."
        assert entries[1].output_length == 100
        assert entries[2].text_input == "Write a Python function for fibonacci."
        assert entries[2].output_length == 150
        print("PASSED: test_dataset_trace_reader_basic")
    finally:
        temp_path.unlink()


def test_dataset_trace_reader_without_output_length() -> None:
    """Test JSONL parsing when output_length is omitted."""
    content = """{"text_input": "What is the capital of France?"}
{"text_input": "Another prompt without output length"}
"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        f.write(content)
        f.flush()
        temp_path = Path(f.name)

    try:
        reader = DatasetTraceReader()
        entries = reader.load_entries(temp_path)

        assert len(entries) == 2, f"Expected 2 entries, got {len(entries)}"
        assert entries[0].text_input == "What is the capital of France?"
        assert entries[0].output_length is None
        assert entries[1].text_input == "Another prompt without output length"
        assert entries[1].output_length is None
        print("PASSED: test_dataset_trace_reader_without_output_length")
    finally:
        temp_path.unlink()


def test_dataset_trace_reader_mixed() -> None:
    """Test JSONL parsing with mixed entries (some with output_length, some without)."""
    content = """{"text_input": "Prompt with length", "output_length": 50}
{"text_input": "Prompt without length"}
{"text_input": "Another with length", "output_length": 200}
"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        f.write(content)
        f.flush()
        temp_path = Path(f.name)

    try:
        reader = DatasetTraceReader()
        entries = reader.load_entries(temp_path)

        assert len(entries) == 3, f"Expected 3 entries, got {len(entries)}"
        assert entries[0].output_length == 50
        assert entries[1].output_length is None
        assert entries[2].output_length == 200
        print("PASSED: test_dataset_trace_reader_mixed")
    finally:
        temp_path.unlink()


def test_dataset_trace_reader_stream_entries() -> None:
    """Test streaming entries from JSONL file."""
    content = """{"text_input": "First prompt", "output_length": 10}
{"text_input": "Second prompt", "output_length": 20}
"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        f.write(content)
        f.flush()
        temp_path = Path(f.name)

    try:
        reader = DatasetTraceReader()
        entries = list(reader.stream_entries(temp_path))

        assert len(entries) == 2, f"Expected 2 entries, got {len(entries)}"
        assert entries[0].text_input == "First prompt"
        assert entries[0].output_length == 10
        assert entries[1].text_input == "Second prompt"
        assert entries[1].output_length == 20
        print("PASSED: test_dataset_trace_reader_stream_entries")
    finally:
        temp_path.unlink()


def test_dataset_trace_reader_skips_empty_lines() -> None:
    """Test that empty lines are skipped."""
    content = """{"text_input": "First prompt"}

{"text_input": "Second prompt"}

"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        f.write(content)
        f.flush()
        temp_path = Path(f.name)

    try:
        reader = DatasetTraceReader()
        entries = reader.load_entries(temp_path)

        assert len(entries) == 2, f"Expected 2 entries, got {len(entries)}"
        print("PASSED: test_dataset_trace_reader_skips_empty_lines")
    finally:
        temp_path.unlink()


def test_dataset_trace_reader_handles_invalid_json() -> None:
    """Test that invalid JSON lines are skipped with a warning."""
    content = """{"text_input": "Valid prompt"}
{invalid json here}
{"text_input": "Another valid prompt"}
"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        f.write(content)
        f.flush()
        temp_path = Path(f.name)

    try:
        reader = DatasetTraceReader()
        entries = reader.load_entries(temp_path)

        # Should skip the invalid line and parse 2 valid entries
        assert len(entries) == 2, f"Expected 2 entries, got {len(entries)}"
        assert entries[0].text_input == "Valid prompt"
        assert entries[1].text_input == "Another valid prompt"
        print("PASSED: test_dataset_trace_reader_handles_invalid_json")
    finally:
        temp_path.unlink()


def test_dataset_trace_reader_handles_missing_text_input() -> None:
    """Test that entries missing text_input field are skipped."""
    content = """{"text_input": "Valid prompt"}
{"output_length": 50}
{"text_input": "Another valid prompt"}
"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        f.write(content)
        f.flush()
        temp_path = Path(f.name)

    try:
        reader = DatasetTraceReader()
        entries = reader.load_entries(temp_path)

        # Should skip the entry without text_input
        assert len(entries) == 2, f"Expected 2 entries, got {len(entries)}"
        print("PASSED: test_dataset_trace_reader_handles_missing_text_input")
    finally:
        temp_path.unlink()


def test_dataset_trace_datagen_completion_api() -> None:
    """Test DatasetTraceDataGenerator with Completion API."""
    content = """{"text_input": "What is the capital of France?", "output_length": 20}
{"text_input": "Explain quantum computing.", "output_length": 100}
"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        f.write(content)
        f.flush()
        temp_path = Path(f.name)

    try:
        api_config = APIConfig(type=APIType.Completion)
        trace_config = TraceConfig(file=str(temp_path), format=TraceFormat.DATASET_TRACE)
        data_config = DataConfig(type=DataGenType.DatasetTrace, trace=trace_config)

        datagen = DatasetTraceDataGenerator(api_config=api_config, config=data_config)

        data_generator = datagen.get_data()
        lazy_data_list = [next(data_generator) for _ in range(2)]
        assert all(isinstance(data, LazyLoadInferenceAPIData) for data in lazy_data_list)

        data_list = [LazyLoadDataMixin.get_request(datagen, data) for data in lazy_data_list]

        assert len(data_list) == 2
        assert isinstance(data_list[0], CompletionAPIData)
        assert data_list[0].prompt == "What is the capital of France?"
        assert data_list[0].max_tokens == 20
        assert isinstance(data_list[1], CompletionAPIData)
        assert data_list[1].prompt == "Explain quantum computing."
        assert data_list[1].max_tokens == 100
        print("PASSED: test_dataset_trace_datagen_completion_api")
    finally:
        temp_path.unlink()


def test_dataset_trace_datagen_chat_api() -> None:
    """Test DatasetTraceDataGenerator with Chat API."""
    content = """{"text_input": "What is the capital of France?", "output_length": 20}
{"text_input": "Explain quantum computing.", "output_length": 100}
"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        f.write(content)
        f.flush()
        temp_path = Path(f.name)

    try:
        api_config = APIConfig(type=APIType.Chat)
        trace_config = TraceConfig(file=str(temp_path), format=TraceFormat.DATASET_TRACE)
        data_config = DataConfig(type=DataGenType.DatasetTrace, trace=trace_config)

        datagen = DatasetTraceDataGenerator(api_config=api_config, config=data_config)

        data_generator = datagen.get_data()
        lazy_data_list = [next(data_generator) for _ in range(2)]

        data_list = [LazyLoadDataMixin.get_request(datagen, data) for data in lazy_data_list]

        assert len(data_list) == 2
        assert isinstance(data_list[0], ChatCompletionAPIData)
        assert len(data_list[0].messages) == 1
        assert data_list[0].messages[0].role == "user"
        assert data_list[0].messages[0].content == "What is the capital of France?"
        assert data_list[0].max_tokens == 20
        assert isinstance(data_list[1], ChatCompletionAPIData)
        assert data_list[1].messages[0].content == "Explain quantum computing."
        assert data_list[1].max_tokens == 100
        print("PASSED: test_dataset_trace_datagen_chat_api")
    finally:
        temp_path.unlink()


def test_dataset_trace_datagen_without_output_length() -> None:
    """Test DatasetTraceDataGenerator when output_length is not specified."""
    content = """{"text_input": "What is the capital of France?"}
{"text_input": "Explain quantum computing."}
"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        f.write(content)
        f.flush()
        temp_path = Path(f.name)

    try:
        api_config = APIConfig(type=APIType.Completion)
        trace_config = TraceConfig(file=str(temp_path), format=TraceFormat.DATASET_TRACE)
        data_config = DataConfig(type=DataGenType.DatasetTrace, trace=trace_config)

        datagen = DatasetTraceDataGenerator(api_config=api_config, config=data_config)

        data_generator = datagen.get_data()
        lazy_data_list = [next(data_generator) for _ in range(2)]
        data_list = [LazyLoadDataMixin.get_request(datagen, data) for data in lazy_data_list]

        # When output_length is not specified, max_tokens should be 0 (server default)
        assert data_list[0].max_tokens == 0
        assert data_list[1].max_tokens == 0
        print("PASSED: test_dataset_trace_datagen_without_output_length")
    finally:
        temp_path.unlink()


def test_dataset_trace_datagen_cycles_entries() -> None:
    """Test that DatasetTraceDataGenerator cycles through entries."""
    content = """{"text_input": "First prompt", "output_length": 10}
{"text_input": "Second prompt", "output_length": 20}
"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        f.write(content)
        f.flush()
        temp_path = Path(f.name)

    try:
        api_config = APIConfig(type=APIType.Completion)
        trace_config = TraceConfig(file=str(temp_path), format=TraceFormat.DATASET_TRACE)
        data_config = DataConfig(type=DataGenType.DatasetTrace, trace=trace_config)

        datagen = DatasetTraceDataGenerator(api_config=api_config, config=data_config)

        data_generator = datagen.get_data()
        # Get more entries than in the file to test cycling
        lazy_data_list = [next(data_generator) for _ in range(5)]
        data_list = [LazyLoadDataMixin.get_request(datagen, data) for data in lazy_data_list]

        # Should cycle: first, second, first, second, first
        assert data_list[0].prompt == "First prompt"
        assert data_list[1].prompt == "Second prompt"
        assert data_list[2].prompt == "First prompt"
        assert data_list[3].prompt == "Second prompt"
        assert data_list[4].prompt == "First prompt"
        print("PASSED: test_dataset_trace_datagen_cycles_entries")
    finally:
        temp_path.unlink()


def test_dataset_trace_datagen_requires_trace_config() -> None:
    """Test that DatasetTraceDataGenerator raises error when trace config is missing."""
    api_config = APIConfig(type=APIType.Completion)
    data_config = DataConfig(type=DataGenType.DatasetTrace, trace=None)

    try:
        DatasetTraceDataGenerator(api_config=api_config, config=data_config)
        raise AssertionError("Expected ValueError to be raised")
    except ValueError as e:
        assert "requires a trace config" in str(e)
        print("PASSED: test_dataset_trace_datagen_requires_trace_config")


def test_dataset_trace_datagen_requires_correct_format() -> None:
    """Test that DatasetTraceDataGenerator raises error for wrong trace format."""
    content = """{"text_input": "Test prompt"}
"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        f.write(content)
        f.flush()
        temp_path = Path(f.name)

    try:
        api_config = APIConfig(type=APIType.Completion)
        # Use wrong format
        trace_config = TraceConfig(file=str(temp_path), format=TraceFormat.AZURE_PUBLIC_DATASET)
        data_config = DataConfig(type=DataGenType.DatasetTrace, trace=trace_config)

        DatasetTraceDataGenerator(api_config=api_config, config=data_config)
        raise AssertionError("Expected ValueError to be raised")
    except ValueError as e:
        assert "DatasetTrace" in str(e)
        print("PASSED: test_dataset_trace_datagen_requires_correct_format")
    finally:
        temp_path.unlink()


def test_dataset_trace_datagen_empty_file() -> None:
    """Test that DatasetTraceDataGenerator raises error for empty file."""
    content = ""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        f.write(content)
        f.flush()
        temp_path = Path(f.name)

    try:
        api_config = APIConfig(type=APIType.Completion)
        trace_config = TraceConfig(file=str(temp_path), format=TraceFormat.DATASET_TRACE)
        data_config = DataConfig(type=DataGenType.DatasetTrace, trace=trace_config)

        DatasetTraceDataGenerator(api_config=api_config, config=data_config)
        raise AssertionError("Expected ValueError to be raised")
    except ValueError as e:
        assert "No valid entries" in str(e)
        print("PASSED: test_dataset_trace_datagen_empty_file")
    finally:
        temp_path.unlink()


if __name__ == "__main__":
    tests: list[Callable[[], None]] = [
        test_dataset_trace_reader_basic,
        test_dataset_trace_reader_without_output_length,
        test_dataset_trace_reader_mixed,
        test_dataset_trace_reader_stream_entries,
        test_dataset_trace_reader_skips_empty_lines,
        test_dataset_trace_reader_handles_invalid_json,
        test_dataset_trace_reader_handles_missing_text_input,
        test_dataset_trace_datagen_completion_api,
        test_dataset_trace_datagen_chat_api,
        test_dataset_trace_datagen_without_output_length,
        test_dataset_trace_datagen_cycles_entries,
        test_dataset_trace_datagen_requires_trace_config,
        test_dataset_trace_datagen_requires_correct_format,
        test_dataset_trace_datagen_empty_file,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception:
            print(f"FAILED: {test.__name__}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"Tests passed: {passed}/{len(tests)}")
    print(f"Tests failed: {failed}/{len(tests)}")



def test_dataset_trace_reader_unicode_handling():
    """Test handling of Unicode characters in prompts."""
    content = """{"text_input": "¿Qué es la inteligencia artificial? 你好", "output_length": 30}
{"text_input": "Explain émojis: 😀🎉🚀", "output_length": 50}
{"text_input": "Test Cyrillic: Привет мир"}
"""
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False, suffix=".jsonl") as f:
        f.write(content)
        f.flush()
        temp_path = Path(f.name)

    try:
        reader = DatasetTraceReader()
        entries = reader.load_entries(temp_path)

        assert len(entries) == 3
        assert "¿Qué es la inteligencia artificial?" in entries[0].text_input
        assert "你好" in entries[0].text_input
        assert "😀🎉🚀" in entries[1].text_input
        assert "Привет мир" in entries[2].text_input
        print("PASSED: test_dataset_trace_reader_unicode_handling")
    finally:
        temp_path.unlink()


def test_dataset_trace_reader_long_prompts():
    """Test handling of very long prompts."""
    long_prompt = "This is a very long prompt. " * 500  # ~15,000 chars
    content = f'{{"text_input": "{long_prompt}", "output_length": 100}}\n'
    
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        f.write(content)
        f.flush()
        temp_path = Path(f.name)

    try:
        reader = DatasetTraceReader()
        entries = reader.load_entries(temp_path)

        assert len(entries) == 1
        assert len(entries[0].text_input) > 10000
        assert entries[0].output_length == 100
        print("PASSED: test_dataset_trace_reader_long_prompts")
    finally:
        temp_path.unlink()


def test_dataset_trace_reader_zero_output_length():
    """Test handling of zero output_length."""
    content = """{"text_input": "Test prompt", "output_length": 0}
"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        f.write(content)
        f.flush()
        temp_path = Path(f.name)

    try:
        reader = DatasetTraceReader()
        entries = reader.load_entries(temp_path)

        assert len(entries) == 1
        assert entries[0].output_length == 0
        print("PASSED: test_dataset_trace_reader_zero_output_length")
    finally:
        temp_path.unlink()


def test_dataset_trace_reader_negative_output_length():
    """Test handling of negative output_length values."""
    content = """{"text_input": "Test prompt", "output_length": -10}
"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        f.write(content)
        f.flush()
        temp_path = Path(f.name)

    try:
        reader = DatasetTraceReader()
        entries = reader.load_entries(temp_path)

        # Should parse successfully; validation happens elsewhere
        assert len(entries) == 1
        assert entries[0].output_length == -10
        print("PASSED: test_dataset_trace_reader_negative_output_length")
    finally:
        temp_path.unlink()


def test_dataset_trace_reader_string_output_length():
    """Test handling of string output_length that can be converted to int."""
    content = """{"text_input": "Test prompt", "output_length": "100"}
"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        f.write(content)
        f.flush()
        temp_path = Path(f.name)

    try:
        reader = DatasetTraceReader()
        entries = reader.load_entries(temp_path)

        assert len(entries) == 1
        assert entries[0].output_length == 100
        assert isinstance(entries[0].output_length, int)
        print("PASSED: test_dataset_trace_reader_string_output_length")
    finally:
        temp_path.unlink()


def test_dataset_trace_reader_invalid_output_length_type():
    """Test handling of non-numeric output_length."""
    content = """{"text_input": "Test prompt", "output_length": "invalid"}
"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        f.write(content)
        f.flush()
        temp_path = Path(f.name)

    try:
        reader = DatasetTraceReader()
        entries = reader.load_entries(temp_path)

        # Should skip entries with invalid output_length
        assert len(entries) == 0
        print("PASSED: test_dataset_trace_reader_invalid_output_length_type")
    finally:
        temp_path.unlink()


def test_dataset_trace_reader_extra_fields():
    """Test that extra fields in JSON are ignored."""
    content = """{"text_input": "Test", "output_length": 50, "extra_field": "ignored", "another": 123}
"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        f.write(content)
        f.flush()
        temp_path = Path(f.name)

    try:
        reader = DatasetTraceReader()
        entries = reader.load_entries(temp_path)

        assert len(entries) == 1
        assert entries[0].text_input == "Test"
        assert entries[0].output_length == 50
        print("PASSED: test_dataset_trace_reader_extra_fields")
    finally:
        temp_path.unlink()


def test_dataset_trace_reader_empty_text_input():
    """Test handling of empty text_input string."""
    content = """{"text_input": "", "output_length": 50}
"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        f.write(content)
        f.flush()
        temp_path = Path(f.name)

    try:
        reader = DatasetTraceReader()
        entries = reader.load_entries(temp_path)

        # Should accept empty string as valid text_input
        assert len(entries) == 1
        assert entries[0].text_input == ""
        print("PASSED: test_dataset_trace_reader_empty_text_input")
    finally:
        temp_path.unlink()


def test_dataset_trace_reader_whitespace_only_text_input():
    """Test handling of whitespace-only text_input."""
    content = """{"text_input": "   \\n\\t  ", "output_length": 50}
"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        f.write(content)
        f.flush()
        temp_path = Path(f.name)

    try:
        reader = DatasetTraceReader()
        entries = reader.load_entries(temp_path)

        # Should accept whitespace as valid text_input
        assert len(entries) == 1
        assert entries[0].text_input.strip() == ""
        print("PASSED: test_dataset_trace_reader_whitespace_only_text_input")
    finally:
        temp_path.unlink()


def test_dataset_trace_reader_special_characters():
    """Test handling of special characters and escape sequences."""
    content = r'''{"text_input": "Test with \"quotes\" and \\ backslashes", "output_length": 50}
{"text_input": "Line with\nnewline and\ttab", "output_length": 30}
'''
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        f.write(content)
        f.flush()
        temp_path = Path(f.name)

    try:
        reader = DatasetTraceReader()
        entries = reader.load_entries(temp_path)

        assert len(entries) == 2
        assert '"quotes"' in entries[0].text_input
        assert '\\' in entries[0].text_input
        assert '\n' in entries[1].text_input
        assert '\t' in entries[1].text_input
        print("PASSED: test_dataset_trace_reader_special_characters")
    finally:
        temp_path.unlink()


def test_dataset_trace_reader_large_file():
    """Test handling of a large file with many entries."""
    num_entries = 1000
    
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        for i in range(num_entries):
            f.write(f'{{"text_input": "Prompt number {i}", "output_length": {i % 100 + 1}}}\n')
        f.flush()
        temp_path = Path(f.name)

    try:
        reader = DatasetTraceReader()
        entries = reader.load_entries(temp_path)

        assert len(entries) == num_entries
        assert entries[0].text_input == "Prompt number 0"
        assert entries[-1].text_input == f"Prompt number {num_entries - 1}"
        print("PASSED: test_dataset_trace_reader_large_file")
    finally:
        temp_path.unlink()


def test_dataset_trace_reader_caching():
    """Test that entries are cached after first load."""
    content = """{"text_input": "Test prompt"}
"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        f.write(content)
        f.flush()
        temp_path = Path(f.name)

    try:
        reader = DatasetTraceReader()
        entries1 = reader.load_entries(temp_path)
        entries2 = reader.load_entries(temp_path)

        # Should return the same cached object
        assert entries1 is entries2
        print("PASSED: test_dataset_trace_reader_caching")
    finally:
        temp_path.unlink()


def test_dataset_trace_reader_stream_vs_load():
    """Test that streaming and loading produce same results."""
    content = """{"text_input": "First", "output_length": 10}
{"text_input": "Second", "output_length": 20}
{"text_input": "Third"}
"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        f.write(content)
        f.flush()
        temp_path = Path(f.name)

    try:
        reader = DatasetTraceReader()
        loaded_entries = reader.load_entries(temp_path)
        
        # Reset reader for streaming
        reader2 = DatasetTraceReader()
        streamed_entries = list(reader2.stream_entries(temp_path))

        assert len(loaded_entries) == len(streamed_entries)
        for loaded, streamed in zip(loaded_entries, streamed_entries):
            assert loaded.text_input == streamed.text_input
            assert loaded.output_length == streamed.output_length
        print("PASSED: test_dataset_trace_reader_stream_vs_load")
    finally:
        temp_path.unlink()


def test_dataset_trace_datagen_get_request_count():
    """Test get_request_count returns correct number of entries."""
    content = """{"text_input": "Prompt 1"}
{"text_input": "Prompt 2"}
{"text_input": "Prompt 3"}
"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        f.write(content)
        f.flush()
        temp_path = Path(f.name)

    try:
        api_config = APIConfig(type=APIType.Completion)
        trace_config = TraceConfig(file=str(temp_path), format=TraceFormat.DATASET_TRACE)
        data_config = DataConfig(type=DataGenType.DatasetTrace, trace=trace_config)

        datagen = DatasetTraceDataGenerator(api_config=api_config, config=data_config)

        assert datagen.get_request_count() == 3
        print("PASSED: test_dataset_trace_datagen_get_request_count")
    finally:
        temp_path.unlink()


def test_dataset_trace_datagen_get_supported_apis():
    """Test get_supported_apis returns correct API types."""
    content = """{"text_input": "Test"}
"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        f.write(content)
        f.flush()
        temp_path = Path(f.name)

    try:
        api_config = APIConfig(type=APIType.Completion)
        trace_config = TraceConfig(file=str(temp_path), format=TraceFormat.DATASET_TRACE)
        data_config = DataConfig(type=DataGenType.DatasetTrace, trace=trace_config)

        datagen = DatasetTraceDataGenerator(api_config=api_config, config=data_config)

        supported = datagen.get_supported_apis()
        assert APIType.Completion in supported
        assert APIType.Chat in supported
        print("PASSED: test_dataset_trace_datagen_get_supported_apis")
    finally:
        temp_path.unlink()


def test_dataset_trace_datagen_io_distribution_not_supported():
    """Test that IO distribution is not supported."""
    content = """{"text_input": "Test"}
"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        f.write(content)
        f.flush()
        temp_path = Path(f.name)

    try:
        api_config = APIConfig(type=APIType.Completion)
        trace_config = TraceConfig(file=str(temp_path), format=TraceFormat.DATASET_TRACE)
        data_config = DataConfig(type=DataGenType.DatasetTrace, trace=trace_config)

        datagen = DatasetTraceDataGenerator(api_config=api_config, config=data_config)

        assert datagen.is_io_distribution_supported() is False
        print("PASSED: test_dataset_trace_datagen_io_distribution_not_supported")
    finally:
        temp_path.unlink()


def test_dataset_trace_datagen_shared_prefix_not_supported():
    """Test that shared prefix is not supported."""
    content = """{"text_input": "Test"}
"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        f.write(content)
        f.flush()
        temp_path = Path(f.name)

    try:
        api_config = APIConfig(type=APIType.Completion)
        trace_config = TraceConfig(file=str(temp_path), format=TraceFormat.DATASET_TRACE)
        data_config = DataConfig(type=DataGenType.DatasetTrace, trace=trace_config)

        datagen = DatasetTraceDataGenerator(api_config=api_config, config=data_config)

        assert datagen.is_shared_prefix_supported() is False
        print("PASSED: test_dataset_trace_datagen_shared_prefix_not_supported")
    finally:
        temp_path.unlink()


def test_dataset_trace_datagen_mixed_output_lengths():
    """Test handling mixed specified and unspecified output_length."""
    content = """{"text_input": "With length", "output_length": 50}
{"text_input": "Without length"}
{"text_input": "With zero", "output_length": 0}
{"text_input": "With large", "output_length": 2000}
"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        f.write(content)
        f.flush()
        temp_path = Path(f.name)

    try:
        api_config = APIConfig(type=APIType.Completion)
        trace_config = TraceConfig(file=str(temp_path), format=TraceFormat.DATASET_TRACE)
        data_config = DataConfig(type=DataGenType.DatasetTrace, trace=trace_config)

        datagen = DatasetTraceDataGenerator(api_config=api_config, config=data_config)

        data_generator = datagen.get_data()
        lazy_data_list = [next(data_generator) for _ in range(4)]
        data_list = [LazyLoadDataMixin.get_request(datagen, data) for data in lazy_data_list]

        assert data_list[0].max_tokens == 50
        assert data_list[1].max_tokens == 0
        assert data_list[2].max_tokens == 0
        assert data_list[3].max_tokens == 2000
        print("PASSED: test_dataset_trace_datagen_mixed_output_lengths")
    finally:
        temp_path.unlink()


def test_dataset_trace_datagen_chat_multiple_messages():
    """Test that chat API wraps prompts correctly."""
    content = """{"text_input": "First question", "output_length": 30}
{"text_input": "Second question", "output_length": 40}
"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        f.write(content)
        f.flush()
        temp_path = Path(f.name)

    try:
        api_config = APIConfig(type=APIType.Chat)
        trace_config = TraceConfig(file=str(temp_path), format=TraceFormat.DATASET_TRACE)
        data_config = DataConfig(type=DataGenType.DatasetTrace, trace=trace_config)

        datagen = DatasetTraceDataGenerator(api_config=api_config, config=data_config)

        data_generator = datagen.get_data()
        lazy_data_list = [next(data_generator) for _ in range(2)]
        data_list = [LazyLoadDataMixin.get_request(datagen, data) for data in lazy_data_list]

        # Verify messages are properly structured
        for data in data_list:
            assert isinstance(data, ChatCompletionAPIData)
            assert len(data.messages) == 1
            assert data.messages[0].role == "user"
        
        assert data_list[0].messages[0].content == "First question"
        assert data_list[1].messages[0].content == "Second question"
        print("PASSED: test_dataset_trace_datagen_chat_multiple_messages")
    finally:
        temp_path.unlink()


def test_dataset_trace_datagen_completion_prompt_structure():
    """Test completion API prompt structure."""
    content = """{"text_input": "Test prompt with special chars: @#$%", "output_length": 50}
"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        f.write(content)
        f.flush()
        temp_path = Path(f.name)

    try:
        api_config = APIConfig(type=APIType.Completion)
        trace_config = TraceConfig(file=str(temp_path), format=TraceFormat.DATASET_TRACE)
        data_config = DataConfig(type=DataGenType.DatasetTrace, trace=trace_config)

        datagen = DatasetTraceDataGenerator(api_config=api_config, config=data_config)

        data_generator = datagen.get_data()
        lazy_data = next(data_generator)
        data = LazyLoadDataMixin.get_request(datagen, lazy_data)

        assert isinstance(data, CompletionAPIData)
        assert data.prompt == "Test prompt with special chars: @#$%"
        assert data.max_tokens == 50
        print("PASSED: test_dataset_trace_datagen_completion_prompt_structure")
    finally:
        temp_path.unlink()


def test_dataset_trace_datagen_lazy_load_indices():
    """Test that lazy load uses correct indices for cycling."""
    content = """{"text_input": "First", "output_length": 10}
{"text_input": "Second", "output_length": 20}
"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        f.write(content)
        f.flush()
        temp_path = Path(f.name)

    try:
        api_config = APIConfig(type=APIType.Completion)
        trace_config = TraceConfig(file=str(temp_path), format=TraceFormat.DATASET_TRACE)
        data_config = DataConfig(type=DataGenType.DatasetTrace, trace=trace_config)

        datagen = DatasetTraceDataGenerator(api_config=api_config, config=data_config)

        # Test that index wraps correctly using modulo
        lazy_data_0 = LazyLoadInferenceAPIData(data_index=0)
        lazy_data_1 = LazyLoadInferenceAPIData(data_index=1)
        lazy_data_2 = LazyLoadInferenceAPIData(data_index=2)  # Should wrap to 0
        lazy_data_3 = LazyLoadInferenceAPIData(data_index=3)  # Should wrap to 1

        data_0 = datagen.load_lazy_data(lazy_data_0)
        data_1 = datagen.load_lazy_data(lazy_data_1)
        data_2 = datagen.load_lazy_data(lazy_data_2)
        data_3 = datagen.load_lazy_data(lazy_data_3)

        assert data_0.prompt == "First"
        assert data_1.prompt == "Second"
        assert data_2.prompt == "First"
        assert data_3.prompt == "Second"
        print("PASSED: test_dataset_trace_datagen_lazy_load_indices")
    finally:
        temp_path.unlink()


def test_dataset_trace_datagen_nonexistent_file():
    """Test handling of nonexistent trace file."""
    api_config = APIConfig(type=APIType.Completion)
    trace_config = TraceConfig(file="/nonexistent/path/trace.jsonl", format=TraceFormat.DATASET_TRACE)
    data_config = DataConfig(type=DataGenType.DatasetTrace, trace=trace_config)

    try:
        DatasetTraceDataGenerator(api_config=api_config, config=data_config)
        assert False, "Expected FileNotFoundError to be raised"
    except FileNotFoundError:
        print("PASSED: test_dataset_trace_datagen_nonexistent_file")


def test_dataset_trace_datagen_single_entry():
    """Test behavior with exactly one entry."""
    content = """{"text_input": "Single prompt", "output_length": 42}
"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        f.write(content)
        f.flush()
        temp_path = Path(f.name)

    try:
        api_config = APIConfig(type=APIType.Completion)
        trace_config = TraceConfig(file=str(temp_path), format=TraceFormat.DATASET_TRACE)
        data_config = DataConfig(type=DataGenType.DatasetTrace, trace=trace_config)

        datagen = DatasetTraceDataGenerator(api_config=api_config, config=data_config)

        assert datagen.get_request_count() == 1

        data_generator = datagen.get_data()
        lazy_data_list = [next(data_generator) for _ in range(5)]
        data_list = [LazyLoadDataMixin.get_request(datagen, data) for data in lazy_data_list]

        # Should repeat the same entry
        for data in data_list:
            assert data.prompt == "Single prompt"
            assert data.max_tokens == 42
        print("PASSED: test_dataset_trace_datagen_single_entry")
    finally:
        temp_path.unlink()


def test_dataset_trace_reader_float_output_length():
    """Test handling of float output_length (should convert to int)."""
    content = """{"text_input": "Test", "output_length": 50.7}
"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        f.write(content)
        f.flush()
        temp_path = Path(f.name)

    try:
        reader = DatasetTraceReader()
        entries = reader.load_entries(temp_path)

        assert len(entries) == 1
        assert entries[0].output_length == 50
        assert isinstance(entries[0].output_length, int)
        print("PASSED: test_dataset_trace_reader_float_output_length")
    finally:
        temp_path.unlink()


def test_dataset_trace_reader_null_output_length():
    """Test handling of null output_length."""
    content = """{"text_input": "Test", "output_length": null}
"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        f.write(content)
        f.flush()
        temp_path = Path(f.name)

    try:
        reader = DatasetTraceReader()
        entries = reader.load_entries(temp_path)

        assert len(entries) == 1
        assert entries[0].output_length is None
        print("PASSED: test_dataset_trace_reader_null_output_length")
    finally:
        temp_path.unlink()


def test_dataset_trace_datagen_preserves_prompt_whitespace():
    """Test that leading/trailing whitespace in prompts is preserved."""
    content = """{"text_input": "  Leading spaces", "output_length": 10}
{"text_input": "Trailing spaces  ", "output_length": 20}
{"text_input": "  Both  ", "output_length": 30}
"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        f.write(content)
        f.flush()
        temp_path = Path(f.name)

    try:
        api_config = APIConfig(type=APIType.Completion)
        trace_config = TraceConfig(file=str(temp_path), format=TraceFormat.DATASET_TRACE)
        data_config = DataConfig(type=DataGenType.DatasetTrace, trace=trace_config)

        datagen = DatasetTraceDataGenerator(api_config=api_config, config=data_config)

        data_generator = datagen.get_data()
        lazy_data_list = [next(data_generator) for _ in range(3)]
        data_list = [LazyLoadDataMixin.get_request(datagen, data) for data in lazy_data_list]

        assert data_list[0].prompt == "  Leading spaces"
        assert data_list[1].prompt == "Trailing spaces  "
        assert data_list[2].prompt == "  Both  "
        print("PASSED: test_dataset_trace_datagen_preserves_prompt_whitespace")
    finally:
        temp_path.unlink()


def test_dataset_trace_reader_array_in_json():
    """Test that JSON arrays are rejected (must be individual objects per line)."""
    content = """[
  {"text_input": "First", "output_length": 10},
  {"text_input": "Second", "output_length": 20}
]
"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        f.write(content)
        f.flush()
        temp_path = Path(f.name)

    try:
        reader = DatasetTraceReader()
        entries = reader.load_entries(temp_path)

        # Should fail to parse as individual lines and return no entries
        assert len(entries) == 0
        print("PASSED: test_dataset_trace_reader_array_in_json")
    finally:
        temp_path.unlink()


def test_dataset_trace_reader_malformed_nested_json():
    """Test handling of complex nested JSON structures."""
    content = """{"text_input": "Valid prompt", "output_length": 50}
{"text_input": {"nested": "object"}, "output_length": 100}
{"text_input": "Another valid", "output_length": 30}
"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        f.write(content)
        f.flush()
        temp_path = Path(f.name)

    try:
        reader = DatasetTraceReader()
        entries = reader.load_entries(temp_path)

        # Should skip the malformed entry with nested object
        assert len(entries) == 2
        assert entries[0].text_input == "Valid prompt"
        assert entries[1].text_input == "Another valid"
        print("PASSED: test_dataset_trace_reader_malformed_nested_json")
    finally:
        temp_path.unlink()


def test_dataset_trace_datagen_very_large_output_length():
    """Test handling of very large output_length values."""
    content = """{"text_input": "Test", "output_length": 999999}
"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        f.write(content)
        f.flush()
        temp_path = Path(f.name)

    try:
        api_config = APIConfig(type=APIType.Completion)
        trace_config = TraceConfig(file=str(temp_path), format=TraceFormat.DATASET_TRACE)
        data_config = DataConfig(type=DataGenType.DatasetTrace, trace=trace_config)

        datagen = DatasetTraceDataGenerator(api_config=api_config, config=data_config)

        data_generator = datagen.get_data()
        lazy_data = next(data_generator)
        data = LazyLoadDataMixin.get_request(datagen, lazy_data)

        assert data.max_tokens == 999999
        print("PASSED: test_dataset_trace_datagen_very_large_output_length")
    finally:
        temp_path.unlink()


def test_dataset_trace_reader_windows_line_endings():
    """Test handling of Windows-style line endings (CRLF)."""
    content = '{"text_input": "First"}\r\n{"text_input": "Second"}\r\n'
    
    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".jsonl") as f:
        f.write(content.encode('utf-8'))
        f.flush()
        temp_path = Path(f.name)

    try:
        reader = DatasetTraceReader()
        entries = reader.load_entries(temp_path)

        assert len(entries) == 2
        assert entries[0].text_input == "First"
        assert entries[1].text_input == "Second"
        print("PASSED: test_dataset_trace_reader_windows_line_endings")
    finally:
        temp_path.unlink()


def test_dataset_trace_reader_mixed_line_endings():
    """Test handling of mixed Unix and Windows line endings."""
    content = '{"text_input": "Unix"}\n{"text_input": "Windows"}\r\n{"text_input": "Unix2"}\n'
    
    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".jsonl") as f:
        f.write(content.encode('utf-8'))
        f.flush()
        temp_path = Path(f.name)

    try:
        reader = DatasetTraceReader()
        entries = reader.load_entries(temp_path)

        assert len(entries) == 3
        print("PASSED: test_dataset_trace_reader_mixed_line_endings")
    finally:
        temp_path.unlink()


def test_dataset_trace_entry_immutability():
    """Test that DatasetTraceEntry attributes can be accessed correctly."""
    entry = DatasetTraceEntry("Test prompt", 100)
    
    assert entry.text_input == "Test prompt"
    assert entry.output_length == 100
    
    # Verify attributes exist and are correctly typed
    assert isinstance(entry.text_input, str)
    assert isinstance(entry.output_length, int)
    print("PASSED: test_dataset_trace_entry_immutability")


def test_dataset_trace_entry_optional_output_length():
    """Test DatasetTraceEntry with None output_length."""
    entry = DatasetTraceEntry("Test prompt", None)
    
    assert entry.text_input == "Test prompt"
    assert entry.output_length is None
    print("PASSED: test_dataset_trace_entry_optional_output_length")


def test_dataset_trace_datagen_generator_infinite():
    """Test that the generator can produce indefinitely."""
    content = """{"text_input": "Test"}
"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        f.write(content)
        f.flush()
        temp_path = Path(f.name)

    try:
        api_config = APIConfig(type=APIType.Completion)
        trace_config = TraceConfig(file=str(temp_path), format=TraceFormat.DATASET_TRACE)
        data_config = DataConfig(type=DataGenType.DatasetTrace, trace=trace_config)

        datagen = DatasetTraceDataGenerator(api_config=api_config, config=data_config)

        data_generator = datagen.get_data()
        
        # Get 1000 entries to verify infinite generation
        lazy_data_list = [next(data_generator) for _ in range(1000)]
        
        assert len(lazy_data_list) == 1000
        # All should be valid LazyLoadInferenceAPIData
        assert all(isinstance(d, LazyLoadInferenceAPIData) for d in lazy_data_list)
        print("PASSED: test_dataset_trace_datagen_generator_infinite")
    finally:
        temp_path.unlink()


if __name__ == "__main__":
    # Update the test list
    tests = [
        test_dataset_trace_reader_basic,
        test_dataset_trace_reader_without_output_length,
        test_dataset_trace_reader_mixed,
        test_dataset_trace_reader_stream_entries,
        test_dataset_trace_reader_skips_empty_lines,
        test_dataset_trace_reader_handles_invalid_json,
        test_dataset_trace_reader_handles_missing_text_input,
        test_dataset_trace_datagen_completion_api,
        test_dataset_trace_datagen_chat_api,
        test_dataset_trace_datagen_without_output_length,
        test_dataset_trace_datagen_cycles_entries,
        test_dataset_trace_datagen_requires_trace_config,
        test_dataset_trace_datagen_requires_correct_format,
        test_dataset_trace_datagen_empty_file,
        test_dataset_trace_reader_unicode_handling,
        test_dataset_trace_reader_long_prompts,
        test_dataset_trace_reader_zero_output_length,
        test_dataset_trace_reader_negative_output_length,
        test_dataset_trace_reader_string_output_length,
        test_dataset_trace_reader_invalid_output_length_type,
        test_dataset_trace_reader_extra_fields,
        test_dataset_trace_reader_empty_text_input,
        test_dataset_trace_reader_whitespace_only_text_input,
        test_dataset_trace_reader_special_characters,
        test_dataset_trace_reader_large_file,
        test_dataset_trace_reader_caching,
        test_dataset_trace_reader_stream_vs_load,
        test_dataset_trace_datagen_get_request_count,
        test_dataset_trace_datagen_get_supported_apis,
        test_dataset_trace_datagen_io_distribution_not_supported,
        test_dataset_trace_datagen_shared_prefix_not_supported,
        test_dataset_trace_datagen_mixed_output_lengths,
        test_dataset_trace_datagen_chat_multiple_messages,
        test_dataset_trace_datagen_completion_prompt_structure,
        test_dataset_trace_datagen_lazy_load_indices,
        test_dataset_trace_datagen_nonexistent_file,
        test_dataset_trace_datagen_single_entry,
        test_dataset_trace_reader_float_output_length,
        test_dataset_trace_reader_null_output_length,
        test_dataset_trace_datagen_preserves_prompt_whitespace,
        test_dataset_trace_reader_array_in_json,
        test_dataset_trace_reader_malformed_nested_json,
        test_dataset_trace_datagen_very_large_output_length,
        test_dataset_trace_reader_windows_line_endings,
        test_dataset_trace_reader_mixed_line_endings,
        test_dataset_trace_entry_immutability,
        test_dataset_trace_entry_optional_output_length,
        test_dataset_trace_datagen_generator_infinite,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAILED: {test.__name__}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"Tests passed: {passed}/{len(tests)}")
    print(f"Tests failed: {failed}/{len(tests)}")

