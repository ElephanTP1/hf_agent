from utils.ollama import OllamaLLM
from langsmith import traceable
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info("Local OLLAMA Test")


class OllamaTests:
    def __init__(self):
        self.ollama = OllamaLLM()
        self.llm = self.ollama.get_llm()
        self.embedding = self.ollama.get_embedding()

    @traceable
    def llm_test(self):
        """Test LLM functionality with translation task"""
        messages = [
            (
                "system",
                "You are a helpful assistant that translates English to French. Translate the user sentence.",
            ),
            ("human", "I love programming."),
        ]
        ai_msg = self.llm.invoke(messages)
        return ai_msg

    def embedding_test(self):
        """Test embedding functionality"""
        query = "I love programming."
        embedding_result = self.embedding.embed_query(query)
        return embedding_result

    def run_llm_test(self):
        """Run and display LLM test results"""
        logger.info("=" * 50)
        logger.info("Testing LLM (Translation)...")
        logger.info("=" * 50)
        try:
            result = self.llm_test()
            logger.info("✅ LLM Test Result:")
            logger.info(f"Response: {result}")
            logger.info(f"Type: {type(result)}")
            return True
        except Exception as e:
            logger.error(f"❌ LLM Test Failed: {e}")
            return False

    def run_embedding_test(self):
        """Run and display embedding test results"""
        logger.info("=" * 50)
        logger.info("Testing Embeddings...")
        logger.info("=" * 50)
        try:
            result = self.embedding_test()
            logger.info("✅ Embedding Test Result:")
            logger.info(f"Embedding dimensions: {len(result)}")
            logger.info(f"First 5 values: {result[:5]}")
            logger.info(f"Type: {type(result)}")
            return True
        except Exception as e:
            logger.error(f"❌ Embedding Test Failed: {e}")
            return False

    def run_all_tests(self):
        """Run all tests and provide summary"""
        logger.info("🚀 Starting Ollama Tests...")

        llm_success = self.run_llm_test()
        embedding_success = self.run_embedding_test()
        
        logger.info("=" * 50)
        logger.info("Test Summary")
        logger.info("=" * 50)
        logger.info(f"LLM Test: {'✅ PASSED' if llm_success else '❌ FAILED'}")
        logger.info(f"Embedding Test: {'✅ PASSED' if embedding_success else '❌ FAILED'}")
        
        if llm_success and embedding_success:
            logger.info("🎉 All tests passed!")
        else:
            logger.warning("⚠️ Some tests failed. Check the output above.")

def main():
    """Main function to run tests"""
    tester = OllamaTests()
    tester.run_all_tests()

if __name__ == "__main__":
    main()