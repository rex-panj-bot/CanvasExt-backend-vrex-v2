#!/usr/bin/env python3
"""
Migration Script: Local Storage → Cloud Storage
Uploads existing PDFs from ./uploads to Google Cloud Storage
Exports SQLite chats.db to PostgreSQL
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import sqlite3
from sqlalchemy import create_engine, text

# Load environment variables
load_dotenv()

from utils.storage_manager import StorageManager
from utils.chat_storage import ChatStorage


def migrate_pdfs_to_gcs():
    """Migrate PDF files from ./uploads to GCS"""
    print("\n" + "=" * 60)
    print("📤 MIGRATING PDFs TO GOOGLE CLOUD STORAGE")
    print("=" * 60 + "\n")

    # Check if GCS is configured
    gcs_bucket = os.getenv("GCS_BUCKET_NAME")
    gcs_project = os.getenv("GCS_PROJECT_ID")

    if not gcs_bucket or not gcs_project:
        print("❌ Error: GCS not configured!")
        print("   Please set GCS_BUCKET_NAME and GCS_PROJECT_ID in .env")
        return False

    # Initialize Storage Manager
    try:
        storage_manager = StorageManager(
            bucket_name=gcs_bucket,
            project_id=gcs_project,
            credentials_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        )
        print(f"✅ Connected to GCS bucket: {gcs_bucket}\n")
    except Exception as e:
        print(f"❌ Failed to connect to GCS: {e}")
        return False

    # Check if uploads directory exists
    uploads_dir = Path("./uploads")
    if not uploads_dir.exists():
        print("⚠️  No ./uploads directory found - nothing to migrate")
        return True

    # Get all PDF files
    pdf_files = list(uploads_dir.glob("*.pdf"))

    if not pdf_files:
        print("⚠️  No PDF files found in ./uploads - nothing to migrate")
        return True

    print(f"Found {len(pdf_files)} PDF files to upload\n")

    # Upload each file
    successful = 0
    failed = 0

    for pdf_path in pdf_files:
        filename = pdf_path.name

        # Extract course_id from filename (format: {course_id}_{original_name}.pdf)
        parts = filename.split('_', 1)
        if len(parts) < 2:
            print(f"⚠️  Skipping {filename} (invalid format)")
            failed += 1
            continue

        course_id = parts[0]
        original_filename = parts[1]

        try:
            # Read file content
            with open(pdf_path, 'rb') as f:
                content = f.read()

            # Upload to GCS
            blob_name = storage_manager.upload_pdf(course_id, original_filename, content)
            size_mb = len(content) / (1024 * 1024)

            print(f"✅ Uploaded: {filename} ({size_mb:.2f} MB) → {blob_name}")
            successful += 1

        except Exception as e:
            print(f"❌ Failed to upload {filename}: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Migration complete: {successful} successful, {failed} failed")
    print(f"{'='*60}\n")

    if successful > 0:
        print("💡 Tip: You can now delete the ./uploads directory if migration succeeded")
        print("        (Keep it as backup until you verify everything works in production)\n")

    return failed == 0


def migrate_chats_to_postgres():
    """Migrate chat history from SQLite to PostgreSQL"""
    print("\n" + "=" * 60)
    print("💬 MIGRATING CHAT HISTORY TO POSTGRESQL")
    print("=" * 60 + "\n")

    # Check if PostgreSQL is configured
    database_url = os.getenv("DATABASE_URL")

    if not database_url or not database_url.startswith("postgresql"):
        print("⚠️  PostgreSQL not configured - skipping chat migration")
        print("   Set DATABASE_URL in .env to migrate chats\n")
        return True

    # Check if SQLite database exists
    sqlite_db = Path("./data/chats.db")
    if not sqlite_db.exists():
        print("⚠️  No SQLite database found - nothing to migrate\n")
        return True

    try:
        # Initialize chat storages
        print("Connecting to databases...")
        sqlite_storage = ChatStorage(db_path="./data/chats.db")
        postgres_storage = ChatStorage(database_url=database_url)
        print("✅ Connected to both databases\n")

        # Get all courses with chats from SQLite
        courses = sqlite_storage.get_all_courses_with_chats()

        if not courses:
            print("⚠️  No chat sessions found - nothing to migrate\n")
            return True

        print(f"Found chat history for {len(courses)} courses\n")

        total_sessions = 0
        total_messages = 0

        # Migrate each course's chats
        for course_id in courses:
            chats = sqlite_storage.get_recent_chats(course_id, limit=1000)  # Get all chats
            print(f"Course {course_id}: {len(chats)} sessions")

            for chat in chats:
                session_id = chat['session_id']

                # Get full session with messages
                full_session = sqlite_storage.get_chat_session(session_id)
                if not full_session:
                    continue

                messages = full_session.get('messages', [])

                # Save to PostgreSQL
                success = postgres_storage.save_chat_session(
                    session_id=session_id,
                    course_id=course_id,
                    messages=messages,
                    title=chat.get('title')
                )

                if success:
                    total_sessions += 1
                    total_messages += len(messages)
                    print(f"  ✅ Migrated session {session_id} ({len(messages)} messages)")
                else:
                    print(f"  ❌ Failed to migrate session {session_id}")

        print(f"\n{'='*60}")
        print(f"Migration complete: {total_sessions} sessions, {total_messages} messages")
        print(f"{'='*60}\n")

        print("💡 Tip: Keep ./data/chats.db as backup until you verify PostgreSQL works\n")

        return True

    except Exception as e:
        print(f"❌ Migration failed: {e}\n")
        return False


def main():
    """Run migration"""
    print("\n🚀 CLOUD MIGRATION TOOL")
    print("=" * 60)
    print("This script will migrate your data to cloud storage:\n")
    print("1. PDFs: ./uploads → Google Cloud Storage")
    print("2. Chats: SQLite → PostgreSQL")
    print("\nMake sure you have:")
    print("  • Set up GCS bucket and PostgreSQL database")
    print("  • Updated .env with cloud credentials")
    print("  • Backed up your data")
    print("=" * 60)

    response = input("\nProceed with migration? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("\n❌ Migration cancelled\n")
        return

    # Migrate PDFs
    pdf_success = migrate_pdfs_to_gcs()

    # Migrate chats
    chat_success = migrate_chats_to_postgres()

    # Summary
    print("\n" + "=" * 60)
    print("MIGRATION SUMMARY")
    print("=" * 60)
    print(f"PDFs:  {'✅ Success' if pdf_success else '❌ Failed'}")
    print(f"Chats: {'✅ Success' if chat_success else '❌ Failed'}")
    print("=" * 60 + "\n")

    if pdf_success and chat_success:
        print("🎉 Migration completed successfully!")
        print("\nNext steps:")
        print("1. Test your application with cloud storage")
        print("2. Verify all PDFs and chats are accessible")
        print("3. Once confirmed working, you can delete:")
        print("   - ./uploads directory")
        print("   - ./data/chats.db file")
        print("\nKeep backups until you're 100% sure everything works!\n")
    else:
        print("⚠️  Migration had errors - please check the output above")
        print("   Do NOT delete local files until migration succeeds\n")


if __name__ == "__main__":
    main()
