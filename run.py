# run.py
import sqlite_fix
# Now import and run the main application
import main

if __name__ == '__main__':
    # Call the Flask app.run method to start the server
    port = int(main.os.environ.get('PORT', 8080))
    main.logger.info(f"Starting server on port {port}")
    main.app.run(debug=True, host='0.0.0.0', port=port)
