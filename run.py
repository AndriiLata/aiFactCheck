from app import create_app

app = create_app()

if __name__ == "__main__":
    # Use env vars or a .env file for HOST/PORT in production
    app.run(debug=True)
