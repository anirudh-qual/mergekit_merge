import json
import random

def generate_random_prompt(max_length=64):
    """Generate a random prompt with token length less than max_length"""
    question_starters = [
    "What", "How", "Why", "When", "Where", "Can", "Should", "Is", "Are", "Do",
    "Could", "Would", "Will", "Did", "Does", "May", "Might", "Which", "Whose",
    "In what ways", "To what extent", "What are the reasons", "How exactly",
    "What is the impact of", "How does", "Why does", "What happens when"]

    topics = [
    "science", "history", "technology", "nature", "music", "art", "sports",
    "food", "travel", "space", 
    "mathematics", "physics", "chemistry", "biology", "engineering",
    "psychology", "philosophy", "economics", "culture", "literature",
    "politics", "education", "environment", "health", "medicine",
    "movies", "gaming", "fashion", "animals", "astronomy",
    "climate", "plants", "architecture", "business", "finance",
    "mythology", "geography", "robots", "AI", "machine learning"]

    verbs = [
    "explain", "describe", "tell", "write", "create", "list", "compare",
    "analyze", "summarize", "define",
    "illustrate", "outline", "predict", "evaluate", "break down",
    "interpret", "highlight", "clarify", "argue", "debate",
    "explore", "investigate", "demonstrate", "identify", "categorize",
    "narrate", "formulate", "develop", "review", "examine"]

    adjectives = [
    "interesting", "simple", "brief", "quick", "detailed", "fun",
    "important", "main", "best", "common",
    "complex", "advanced", "basic", "creative", "unique", "practical",
    "insightful", "deep", "short", "comprehensive",
    "engaging", "fascinating", "useful", "challenging", "informative",
    "critical", "essential", "modern", "classic", "trending"]

    
    prompt_type = random.choice(["question", "instruction", "completion"])
    
    if prompt_type == "question":
        starter = random.choice(question_starters)
        verb = random.choice(verbs)
        topic = random.choice(topics)
        adj = random.choice(adjectives)
        prompt = f"{starter} can you {verb} {adj} aspects of {topic}?"
    elif prompt_type == "instruction":
        verb = random.choice(verbs)
        topic = random.choice(topics)
        adj = random.choice(adjectives)
        prompt = f"{verb.capitalize()} {adj} facts about {topic}."
    else:  # completion
        topic = random.choice(topics)
        prompt = f"The most important thing about {topic} is"
    
    # Ensure it's under max_length (rough approximation: ~4 chars per token)
    if len(prompt) > max_length * 4:
        prompt = prompt[:max_length * 4]
    
    return prompt

def generate_dataset(num_prompts=100, output_file="prompts.jsonl", max_length=64):
    """Generate a JSONL dataset file with random prompts"""
    with open(output_file, 'w') as f:
        for _ in range(num_prompts//2):
            prompt = generate_random_prompt(max_length)
            prompt = "<|use_instruct|>" + prompt
            json.dump({"prompt": prompt}, f)
            f.write('\n')
        for _ in range(num_prompts//2):
            prompt = generate_random_prompt(max_length)
            prompt = "<|use_base|>" + prompt
            json.dump({"prompt": prompt}, f)
            f.write('\n')
    
    print(f"Generated {num_prompts} prompts in {output_file}")

if __name__ == "__main__":
    # Generate dataset with 200 prompts
    generate_dataset(num_prompts=4000, output_file="custom_prompts.jsonl")