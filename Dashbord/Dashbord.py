import pandas as pd
import numpy as np
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
import gradio as gr
import os
import sys
import traceback

# Load environment variables
load_dotenv()

# Try to load the books dataset with error handling
try:
    books = pd.read_csv("dataset/books_with_emotions.csv")
    
    # Process thumbnail URLs safely
    books["large_thumbnail"] = books["thumbnail"].apply(
        lambda x: str(x) + "&fife=w800" if pd.notna(x) else None
    )
    
    # Set a default image for missing thumbnails
    default_image = "/home/veinmahzy/Booker/Dashbord/cover-not-found.jpg"
    if not os.path.exists(default_image):
        default_image = "cover-not-found.jpg"
        
    books["large_thumbnail"] = books["large_thumbnail"].fillna(default_image)
    
except Exception as e:
    print(f"Error loading books dataset: {str(e)}")
    # Create an empty DataFrame with the necessary columns if loading fails
    books = pd.DataFrame(columns=["isbn13", "title", "authors", "description", 
                                 "thumbnail", "large_thumbnail", "simple_categories",
                                 "joy", "surprise", "anger", "fear", "sadness"])

# Define a simple no-results image
NO_RESULTS_IMAGE = "no-results-found.jpg"

# Initialize the document database with error handling
try:
    # Load and process text data
    raw_documents = TextLoader("model/tagged_description.txt").load()
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
    documents = text_splitter.split_documents(raw_documents)

    # Initialize HuggingFace embeddings
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create vector database with HuggingFace embeddings
    db_books = Chroma.from_documents(documents, embedding=embedding_model)
    
except Exception as e:
    print(f"Error initializing document database: {str(e)}")
    # Create a dummy db_books object that will return empty results
    class DummyDB:
        def similarity_search(self, query, k=10):
            return []
    db_books = DummyDB()

def safe_get(dictionary, key, default=""):
    """Safely get a value from a dictionary."""
    try:
        value = dictionary.get(key, default)
        return str(value) if pd.notna(value) else default
    except:
        return default

def retrieve_semantic_recommendations(
    query: str,
    category: str = None,
    tone: str = None,
    initial_top_k: int = 50,
    final_top_k: int = 16,
) -> pd.DataFrame:
    """Retrieve semantic recommendations with comprehensive error handling."""
    # Return empty DataFrame for empty queries
    if not query or not query.strip():
        return pd.DataFrame()
    
    try:
        # Get recommendations from the vector database
        recs = db_books.similarity_search(query, k=initial_top_k)
        
        # If no recommendations, return empty DataFrame
        if not recs:
            return pd.DataFrame()
        
        # Extract book IDs safely
        books_list = []
        for rec in recs:
            try:
                content = rec.page_content.strip('"').split()
                if content:
                    books_list.append(int(content[0]))
            except:
                continue
        
        # If no valid book IDs were extracted, return empty DataFrame
        if not books_list:
            return pd.DataFrame()
        
        # Get books that match the extracted IDs
        book_recs = books[books["isbn13"].isin(books_list)]
        
        # If no matching books, return empty DataFrame
        if book_recs.empty:
            return pd.DataFrame()
        
        # Apply category filter if specified
        if category and category != "All":
            filtered_recs = book_recs[book_recs["simple_categories"] == category]
            # If no books match the category, return empty DataFrame
            if not filtered_recs.empty:
                book_recs = filtered_recs
        
        # Sort by emotion if tone is specified
        if tone and tone != "All":
            emotion_map = {
                "Happy": "joy",
                "Surprising": "surprise", 
                "Angry": "anger",
                "Suspenseful": "fear",
                "Sad": "sadness"
            }
            
            # Get corresponding emotion column
            emotion_col = emotion_map.get(tone)
            
            # Sort by emotion if column exists
            if emotion_col and emotion_col in book_recs.columns:
                book_recs = book_recs.sort_values(by=emotion_col, ascending=False)
        
        # Return top recommendations
        return book_recs.head(final_top_k)
        
    except Exception as e:
        print(f"Error in recommendation engine: {str(e)}")
        traceback.print_exc()
        return pd.DataFrame()

def create_no_results_message(query, category, tone):
    """Create a descriptive message for when no results are found."""
    message = f"No books found matching: '{query}'"
    
    if category and category != "All":
        message += f", category: '{category}'"
    
    if tone and tone != "All":
        message += f", tone: '{tone}'"
        
    message += ". Please try different search criteria."
    return message

def recommend_books(query, category, tone):
    """Main recommendation function with comprehensive error handling."""
    try:
        # Get recommendations
        recommendations = retrieve_semantic_recommendations(query, category, tone)
        
        # If no recommendations, return a helpful message
        if recommendations.empty:
            message = create_no_results_message(query, category, tone)
            return [(NO_RESULTS_IMAGE, message)]
        
        # Process recommendations
        results = []
        for _, row in recommendations.iterrows():
            try:
                # Get book details safely
                title = safe_get(row, "title", "Untitled")
                authors = safe_get(row, "authors", "Unknown Author")
                description = safe_get(row, "description", "")
                
                # Format authors
                authors_split = authors.split(";")
                if len(authors_split) == 2:
                    authors_str = f"{authors_split[0]} and {authors_split[1]}"
                elif len(authors_split) > 2:
                    authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
                else:
                    authors_str = authors
                
                # Truncate description
                truncated_desc = " ".join(description.split()[:30]) + "..." if description else ""
                
                # Create caption
                caption = f"{title} by {authors_str}"
                if truncated_desc:
                    caption += f": {truncated_desc}"
                
                # Get thumbnail
                thumbnail = safe_get(row, "large_thumbnail", NO_RESULTS_IMAGE)
                
                # Add to results
                results.append((thumbnail, caption))
                
            except Exception as e:
                print(f"Error processing book result: {str(e)}")
                continue
        
        # If no valid results after processing, return a message
        if not results:
            return [(NO_RESULTS_IMAGE, "No valid book data found. Please try again.")]
            
        return results
        
    except Exception as e:
        print(f"Error in recommend_books: {str(e)}")
        traceback.print_exc()
        return [(NO_RESULTS_IMAGE, "An error occurred while searching. Please try again.")]

# Setup Gradio interface
try:
    # Get unique categories safely
    if not books.empty and "simple_categories" in books.columns:
        unique_categories = sorted(books["simple_categories"].dropna().unique())
    else:
        unique_categories = []
    
    categories = ["All"] + unique_categories
    tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]
    
    def search_books(query, category, tone):
        try:
            results = recommend_books(query, category, tone)
            return results
        except Exception as e:
            print(f"Error in search_books: {str(e)}")
            traceback.print_exc()
            return [(NO_RESULTS_IMAGE, "An error occurred while searching. Please try again.")]
    
    # Create Gradio interface
    with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
        gr.Markdown("# 📘 Cariin: Book Finder")

        with gr.Row():
            user_query = gr.Textbox(
                label="Describe the type of book you're looking for:",
                placeholder="e.g., A story about forgiveness"
            )
            category_dropdown = gr.Dropdown(
                choices=categories,
                label="Choose a category:",
                value="All"
            )
            tone_dropdown = gr.Dropdown(
                choices=tones,
                label="Select emotional tone:",
                value="All"
            )
            submit_button = gr.Button("🔍 Search Books")

        status_text = gr.Markdown("")

        gr.Markdown("## 📚 Book Recommendations")
        output = gr.Gallery(label="Recommended Books", columns=8, rows=2)

        # Function with status updates
        def search_books_with_status(query, category, tone):
            status_text.update(value="🔄 Loading recommendations...")
            results = recommend_books(query, category, tone)
            status_text.update(value="✅ Done!" if results else "⚠️ No books found.")
            return results

        # Chain button click to both the loading message and search function
        submit_button.click(
            fn=lambda: gr.update(value="🔄 Loading recommendations..."),
            inputs=[],
            outputs=[status_text]
        ).then(
            fn=search_books_with_status,
            inputs=[user_query, category_dropdown, tone_dropdown],
            outputs=[output]
        )

    if __name__ == "__main__":
        dashboard.launch()
        
except Exception as e:
    print(f"Error setting up Gradio interface: {str(e)}")
    traceback.print_exc()