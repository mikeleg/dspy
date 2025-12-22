"""
Tests and Examples for Chain of Density Module

This file demonstrates how to use the ChainOfDensity module and includes
tests to verify its functionality.
"""

import dspy

# Import our module (in the actual implementation, this would be from dspy)
from chain_of_density import (
    ChainOfDensity,
    ChainOfDensityWithPreference,
    chain_of_density,
)


# ============================================================================
# Example Article for Testing
# ============================================================================

SAMPLE_ARTICLE = """
Apple Inc. announced its latest iPhone 14 today at a major press event held at 
the Steve Jobs Theater in Cupertino, California. CEO Tim Cook took the stage to 
highlight the new features, including a significantly improved 48-megapixel camera 
system, satellite connectivity for emergency SOS messages, and a new crash detection 
feature using advanced accelerometers and gyroscopes.

The device will be available in four variants: iPhone 14, iPhone 14 Plus, iPhone 14 Pro, 
and iPhone 14 Pro Max. The base model starts at $799, while the Pro Max tops out at $1,099.
Pre-orders begin September 9th, with shipping starting September 16th.

Apple's VP of Worldwide Marketing, Greg Joswiak, emphasized that the Pro models feature 
the new A16 Bionic chip, manufactured using a 4-nanometer process, making it the fastest 
chip ever in a smartphone. The always-on display, a first for iPhone, was also unveiled,
drawing comparisons to the Apple Watch.

Industry analysts from firms including Morgan Stanley and Wedbush Securities have 
predicted strong sales, with estimates suggesting Apple could sell 90 million units 
in the first quarter alone. The satellite connectivity feature, developed in partnership 
with Globalstar, represents a $450 million investment and could prove crucial for 
users in remote areas.
"""

NEWS_ARTICLE = """
The European Central Bank raised interest rates by 75 basis points on Thursday, 
the largest increase in its history, as President Christine Lagarde signaled that 
further hikes are likely in the coming months to combat record-high inflation across 
the eurozone.

The decision brings the main refinancing rate to 1.25%, up from 0.50%, marking the 
second consecutive rate rise this year after more than a decade of historically low 
or negative rates. The deposit facility rate, which had been negative since 2014, 
now stands at 0.75%.

Inflation in the 19 countries sharing the euro currency hit 9.1% in August, more 
than four times the ECB's 2% target. Energy prices, driven by Russia's reduction 
of natural gas supplies following its invasion of Ukraine, have been the primary 
driver, but core inflation excluding food and energy also rose to 4.3%.

Lagarde, speaking at a press conference in Frankfurt, acknowledged that the rate 
increases would slow economic growth but emphasized that price stability remains 
the central bank's primary mandate. "We expect to raise interest rates further," 
she stated, while declining to specify the size of future increases.

Financial markets reacted positively to the announcement, with the euro gaining 
0.8% against the dollar to reach $1.0024. European bank stocks rose sharply, with 
Deutsche Bank and BNP Paribas both gaining over 3%.
"""


# ============================================================================
# Basic Usage Examples
# ============================================================================

def example_basic_usage():
    """Basic example of using ChainOfDensity."""
    print("=" * 80)
    print("EXAMPLE: Basic Chain of Density Usage")
    print("=" * 80)
    
    # Configure DSPy with your preferred LM
    # dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
    
    # Create the Chain of Density module
    cod = ChainOfDensity(
        num_steps=5,        # Number of densification iterations
        target_length=80,   # Target summary length in words
    )
    
    # Generate summaries
    result = cod(article=SAMPLE_ARTICLE)
    
    # Print results
    print("\n📊 Summary Progression (Sparse → Dense):\n")
    for i, summary in enumerate(result.summaries):
        density_info = result.entity_density[i]
        print(f"Step {i} (density: {density_info['density']:.3f}):")
        print(f"  {summary}")
        print()
        
        if i > 0:
            print(f"  Entities added: {result.entities_added[i-1]}")
            print()
    
    print("\n✅ Final Summary:")
    print(f"  {result.final_summary}")
    
    return result


def example_preferred_density():
    """Example using ChainOfDensityWithPreference to select optimal density."""
    print("=" * 80)
    print("EXAMPLE: Chain of Density with Preferred Step")
    print("=" * 80)
    
    # According to the paper, step 3-4 out of 5 is often preferred by humans
    cod = ChainOfDensityWithPreference(
        num_steps=5,
        target_length=80,
        preferred_step=3,  # Get moderately dense summary
    )
    
    result = cod(article=NEWS_ARTICLE)
    
    print(f"\n📊 Selected step {result.preferred_step} as the preferred density")
    print(f"\n✅ Preferred Summary:")
    print(f"  {result.final_summary}")
    
    return result


def example_custom_selection():
    """Example with custom selection function."""
    print("=" * 80)
    print("EXAMPLE: Chain of Density with Custom Selection")
    print("=" * 80)
    
    # Custom function that selects based on entity count target
    def select_by_entity_count(summaries, entities_added):
        """Select the summary with approximately 8 new entities total."""
        total_entities = 0
        for i, entities in enumerate(entities_added):
            total_entities += len(entities)
            if total_entities >= 8:
                return i + 1  # +1 because step 0 is initial summary
        return len(summaries) - 1
    
    cod = ChainOfDensityWithPreference(
        num_steps=5,
        target_length=80,
        selection_fn=select_by_entity_count,
    )
    
    result = cod(article=SAMPLE_ARTICLE)
    
    print(f"\n📊 Custom selection chose step {result.preferred_step}")
    print(f"\n✅ Selected Summary:")
    print(f"  {result.final_summary}")
    
    return result


def example_quick_function():
    """Example using the convenience function."""
    print("=" * 80)
    print("EXAMPLE: Quick Chain of Density Function")
    print("=" * 80)
    
    result = chain_of_density(
        article=SAMPLE_ARTICLE,
        num_steps=3,
        target_length=60,
    )
    
    print(f"\n✅ Final Summary:")
    print(f"  {result.final_summary}")
    
    return result


# ============================================================================
# Integration Test
# ============================================================================

def test_chain_of_density_structure():
    """Test that the module produces correct output structure."""
    print("=" * 80)
    print("TEST: Verifying Output Structure")
    print("=" * 80)
    
    cod = ChainOfDensity(num_steps=3, target_length=50)
    
    # We can test structure even without an actual LM by mocking
    # For real testing, you would configure an LM
    
    # Test that module has correct attributes
    assert hasattr(cod, 'num_steps')
    assert hasattr(cod, 'target_length')
    assert hasattr(cod, 'initial_summarizer')
    assert hasattr(cod, 'densifier')
    
    assert cod.num_steps == 3
    assert cod.target_length == 50
    
    print("✅ Module structure is correct")
    print("✅ All attributes present")
    
    return True


def test_signatures():
    """Test that signatures are properly defined."""
    print("=" * 80)
    print("TEST: Verifying Signatures")
    print("=" * 80)
    
    from chain_of_density import (
        InitialSummary,
        DensifySummary,
        DensificationStep,
        IdentifyMissingEntities,
    )
    
    # Check InitialSummary
    assert 'article' in InitialSummary.model_fields
    assert 'target_length' in InitialSummary.model_fields
    assert 'summary' in InitialSummary.model_fields
    print("✅ InitialSummary signature is correct")
    
    # Check DensificationStep
    assert 'article' in DensificationStep.model_fields
    assert 'previous_summary' in DensificationStep.model_fields
    assert 'missing_entities' in DensificationStep.model_fields
    assert 'denser_summary' in DensificationStep.model_fields
    print("✅ DensificationStep signature is correct")
    
    return True


# ============================================================================
# Performance Comparison
# ============================================================================

def compare_vanilla_vs_cod():
    """Compare vanilla summarization with Chain of Density."""
    print("=" * 80)
    print("COMPARISON: Vanilla vs Chain of Density")
    print("=" * 80)
    
    # This would require an actual LM configured
    # Here we show the structure of such a comparison
    
    class VanillaSummarize(dspy.Signature):
        """Summarize the article in about 80 words."""
        article: str = dspy.InputField()
        summary: str = dspy.OutputField()
    
    vanilla = dspy.Predict(VanillaSummarize)
    cod = ChainOfDensity(num_steps=5, target_length=80)
    
    print("""
    In a real comparison, you would:
    1. Run both on the same articles
    2. Compare entity density (entities per token)
    3. Compare abstractiveness (n-gram overlap with source)
    4. Compare human preferences
    
    Expected results based on the paper:
    - CoD summaries have ~2x more entities per token
    - CoD summaries are more abstractive
    - CoD summaries show less "lead bias" (don't just copy the beginning)
    - Humans prefer step 3-4 summaries (moderate density)
    """)


# ============================================================================
# Main
# ============================================================================

def main():
    """Run all examples and tests."""
    print("\n" + "🔗 CHAIN OF DENSITY MODULE DEMO ".center(80, "=") + "\n")
    
    # Run structural tests (don't require LM)
    test_chain_of_density_structure()
    print()
    test_signatures()
    print()
    
    # Show comparison info
    compare_vanilla_vs_cod()
    print()
    
    print("=" * 80)
    print("NOTE: To run the actual examples, configure a language model:")
    print()
    print("  import dspy")
    print('  dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))')
    print()
    print("Then run:")
    print("  example_basic_usage()")
    print("  example_preferred_density()")
    print("  example_custom_selection()")
    print("=" * 80)


if __name__ == "__main__":
    main()
