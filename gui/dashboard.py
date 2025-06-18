import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import customtkinter as ctk
import tkinter as tk
from tkinter import scrolledtext
from gmailapi.gmail_connect import authenticate_gmail, fetch_emails, get_email_by_id
import webbrowser
import threading

# Import spam prediction model from the root directory
# Changed from spam_prediction.predict import gui_predict_spam
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # Ensure root directory is in path
from logisticregression import predict_spam  # Adjust function name 

# Initialize the main app
ctk.set_appearance_mode("Light")  # Options: "System" (default), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Default theme

class SpamPredictionApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Email Spam Detection")
        self.geometry("1000x700")
        
        # Store email data and Gmail service
        self.emails = []
        self.spam_emails = []
        self.gmail_service = None
        self.selected_email_id = None
        self.current_email_content = None  # To store the content of the currently displayed email

        self.viewed_email_ids = set()  # To track which emails have been viewed
        
        # Create a status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")

        # Create a PanedWindow
        self.paned_window = tk.PanedWindow(self, orient="horizontal")
        self.paned_window.pack(fill="both", expand=True, pady=(0, 25))  # Leave space at bottom for status

        # Left Sidebar
        self.sidebar_frame = ctk.CTkFrame(self.paned_window, width=235)
        self.paned_window.add(self.sidebar_frame)
        self.sidebar_frame.pack_propagate(False)  # Prevent the frame from resizing based on its contents

        # Add this near the sidebar creation, replace the simple header label
        self.header_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.header_frame.pack(pady=(15, 20), padx=10, fill="x")

        # You can add an icon SVG or use a Unicode character
        self.app_icon = ctk.CTkLabel(
            self.header_frame, 
            text="📧", 
            font=ctk.CTkFont(size=22)
        )
        self.app_icon.pack(side="left", padx=(5, 0))

        self.header_label = ctk.CTkLabel(
            self.header_frame, 
            text="Email Spam Detection", 
            font=ctk.CTkFont(size=14, weight="bold"), 
            anchor="w", 
            justify="left"
        )
        self.header_label.pack(side="left", padx=10)

        # Add a gradient line below header
        self.gradient_frame = ctk.CTkFrame(self.sidebar_frame, height=2, fg_color=("#3a7ebf", "#1f538d"))
        self.gradient_frame.pack(fill="x", padx=10, pady=(0, 15))

        # Folders section
        self.folders_label = ctk.CTkLabel(
            self.sidebar_frame, 
            text="Folders", 
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="w"
        )
        self.folders_label.pack(anchor="w", padx=15, pady=(15, 5))

        # Inbox folder
        self.inbox_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent", height=30)
        self.inbox_frame.pack(fill="x", padx=10, pady=2)

        self.inbox_icon = ctk.CTkLabel(
            self.inbox_frame,
            text="📥", 
            font=ctk.CTkFont(size=16),
            width=20
        )
        self.inbox_icon.pack(side="left", padx=(5, 0))

        self.inbox_label = ctk.CTkLabel(
            self.inbox_frame,
            text="Inbox",
            anchor="w"
        )
        self.inbox_label.pack(side="left", padx=5, fill="x", expand=True)

        # Spam folder with click action
        self.spam_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent", height=30)
        self.spam_frame.pack(fill="x", padx=10, pady=2)

        self.spam_icon = ctk.CTkLabel(
            self.spam_frame,
            text="⚠️", 
            font=ctk.CTkFont(size=16),
            width=20
        )
        self.spam_icon.pack(side="left", padx=(5, 0))

        self.spam_label = ctk.CTkLabel(
            self.spam_frame,
            text="Spam",
            anchor="w"
        )
        self.spam_label.pack(side="left", padx=5, fill="x", expand=True)

        # Make the spam folder clickable to show spam emails
        for widget in [self.spam_frame, self.spam_icon, self.spam_label]:
            widget.bind("<Button-1>", self.show_spam_folder)
            widget.bind("<Enter>", lambda e, w=widget: self.on_folder_hover_enter(w))
            widget.bind("<Leave>", lambda e, w=widget: self.on_folder_hover_leave(w))

        # Make the inbox folder clickable to show all emails
        for widget in [self.inbox_frame, self.inbox_icon, self.inbox_label]:
            widget.bind("<Button-1>", self.show_inbox_folder)
            widget.bind("<Enter>", lambda e, w=widget: self.on_folder_hover_enter(w))
            widget.bind("<Leave>", lambda e, w=widget: self.on_folder_hover_leave(w))

        # Add a separator after folders
        self.folder_separator = ctk.CTkFrame(self.sidebar_frame, height=1, fg_color=("gray80", "gray40"))
        self.folder_separator.pack(fill="x", padx=15, pady=(10, 15))

        # Fetch Emails button
        self.fetch_button = ctk.CTkButton(
            self.sidebar_frame, 
            text="Fetch Emails", 
            command=self.start_fetch_thread,
            corner_radius=8,
            hover_color=("#3a7ebf", "#1f538d"),
            border_width=0,
            height=38
        )
        self.fetch_button.pack(pady=10, padx=15, fill="x")
        
        # Setup Gmail API button
        self.setup_button = ctk.CTkButton(
            self.sidebar_frame, 
            text="Setup Gmail API", 
            command=self.open_google_cloud,
            corner_radius=8,
            hover_color=("#3a7ebf", "#1f538d"),
            border_width=0,
            height=38
        )
        self.setup_button.pack(pady=10, padx=15, fill="x")
        
        # Predict Spam button - connect to prediction function
        self.predict_button = ctk.CTkButton(
            self.sidebar_frame, 
            text="Detect Spam", 
            command=self.predict_spam, 
            state="disabled",
            corner_radius=8,
            hover_color=("#3a7ebf", "#1f538d"),
            border_width=0,
            height=38
        )
        self.predict_button.pack(pady=10, padx=15, fill="x")

        # Add a frame for prediction results
        self.prediction_frame = ctk.CTkFrame(
            self.sidebar_frame,
            corner_radius=10,
            border_width=1,
            border_color=("gray70", "gray40")
        )
        self.prediction_frame.pack(pady=15, padx=15, fill="x")
        
        self.prediction_label = ctk.CTkLabel(self.prediction_frame, text="Detection: Not analyzed", 
                                           font=ctk.CTkFont(weight="bold"))
        self.prediction_label.pack(pady=5, anchor="w")
        
        self.confidence_label = ctk.CTkLabel(self.prediction_frame, text="Confidence: 0%")
        self.confidence_label.pack(pady=5, anchor="w")

        # Main Content Frame with Paned Window for email list and email detail
        self.main_content_frame = ctk.CTkFrame(self.paned_window)
        self.paned_window.add(self.main_content_frame)
        
        self.content_paned = tk.PanedWindow(self.main_content_frame, orient="horizontal")
        self.content_paned.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Email List Frame (Left Side of Main Content)
        self.email_list_frame = ctk.CTkFrame(self.content_paned)
        self.content_paned.add(self.email_list_frame, width=350)
        
        # Search Bar in Email List Frame
        self.search_frame = ctk.CTkFrame(self.email_list_frame)
        self.search_frame.pack(fill="x", padx=5, pady=5)
        
        # Search Entry
        self.search_entry = ctk.CTkEntry(
            self.search_frame, 
            placeholder_text="Search...",
            height=32,
            corner_radius=8,
            border_width=1
        )
        self.search_entry.pack(side="left", padx=5, fill="x", expand=True)
        
        # Search Button
        self.search_button = ctk.CTkButton(self.search_frame, text="Search", width=80)
        self.search_button.pack(side="right", padx=5)
        
        # Email List Canvas with Scrollbar
        self.list_container = ctk.CTkFrame(self.email_list_frame)
        self.list_container.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create a canvas for scrolling
        self.canvas = tk.Canvas(self.list_container, bg=self.cget("fg_color")[1 if ctk.get_appearance_mode() == "Dark" else 0])
        self.scrollbar = ctk.CTkScrollbar(self.list_container, orientation="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        
        # Frame inside canvas for email items
        self.email_list_inner = ctk.CTkFrame(self.canvas)
        self.canvas_frame = self.canvas.create_window((0, 0), window=self.email_list_inner, anchor="nw")
        
        self.email_list_inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", self.resize_canvas_frame)
        
        # Email Detail Frame (Right Side of Main Content)
        self.email_detail_frame = ctk.CTkFrame(self.content_paned)
        self.content_paned.add(self.email_detail_frame)
        
        # Email detail header
        self.detail_header = ctk.CTkFrame(self.email_detail_frame, fg_color="transparent")
        self.detail_header.pack(fill="x", padx=15, pady=10)
        
        self.subject_label = ctk.CTkLabel(
            self.detail_header, 
            text="", 
            font=ctk.CTkFont(size=18, weight="bold"),
            wraplength=400
        )
        self.subject_label.pack(anchor="w", padx=5, pady=5)
        
        self.from_label = ctk.CTkLabel(self.detail_header, text="")
        self.from_label.pack(anchor="w", padx=5, pady=2)
        
        self.to_label = ctk.CTkLabel(self.detail_header, text="")
        self.to_label.pack(anchor="w", padx=5, pady=2)
        
        self.date_label = ctk.CTkLabel(self.detail_header, text="")
        self.date_label.pack(anchor="w", padx=5, pady=2)
        
        # Separator
        self.separator = ctk.CTkFrame(
            self.email_detail_frame, 
            height=1, 
            fg_color=("gray80", "gray40")
        )
        self.separator.pack(fill="x", padx=15, pady=10)
        
        # Email content
        self.email_content = scrolledtext.ScrolledText(
            self.email_detail_frame, 
            wrap="word",
            font=("Arial", 12),
            padx=10,
            pady=10,
            relief="flat",
            borderwidth=0
        )
        self.email_content.pack(fill="both", expand=True, padx=15, pady=10)

        # Set initial colors based on current mode
        if ctk.get_appearance_mode() == "Dark":
            self.email_content.config(bg='#2b2b2b', fg='white')
        else:
            self.email_content.config(bg='white', fg='black')
        
        # Status bar at the bottom
        self.status_bar = ctk.CTkFrame(self, height=30, fg_color=("gray90", "gray20"))
        self.status_bar.pack(side="bottom", fill="x")

        self.status_label = ctk.CTkLabel(
            self.status_bar, 
            textvariable=self.status_var, 
            anchor="w",
            font=ctk.CTkFont(size=12)
        )
        self.status_label.pack(side="left", padx=15, pady=5)

        # Dark/Light Mode Toggle
        self.mode_toggle = ctk.CTkSwitch(self.sidebar_frame, text="Dark Mode", command=self.toggle_mode)
        self.mode_toggle.pack(side="bottom", pady=10, padx=10, anchor="w")
        
        # Initial setup
        if ctk.get_appearance_mode() == "Dark":
            self.mode_toggle.select()
            
        # Setup search functionality
        self.setup_search_functionality()
            
        # Display welcome message
        self.display_welcome_message()
    
    def setup_search_functionality(self):
        """Setup the search functionality"""
        # Connect the search button to the search function
        self.search_button.configure(command=self.search_emails)
        
        # Add Enter key binding to search entry
        self.search_entry.bind("<Return>", lambda event: self.search_emails())
        
        # Add clear search button
        self.clear_search_button = ctk.CTkButton(self.search_frame, text="×", width=30, 
                                               command=self.clear_search)
        self.clear_search_button.pack(side="right", padx=0)
        self.clear_search_button.configure(state="disabled")  # Initially disabled
    
    def search_emails(self):
        """Search emails based on keyword"""
        search_term = self.search_entry.get().strip().lower()
        
        if not search_term:
            self.update_status("Please enter a search term")
            return
        
        if not self.emails:
            self.update_status("No emails loaded to search")
            return
        
        self.update_status(f"Searching for '{search_term}'...")
        
        # Enable clear search button
        self.clear_search_button.configure(state="normal")
        
        # Clear previous email list
        for widget in self.email_list_inner.winfo_children():
            widget.destroy()
        
        # Filter emails
        filtered_emails = []
        for email in self.emails:
            # Search in subject, sender, and snippet
            if (search_term in email['subject'].lower() or 
                search_term in email['sender'].lower() or 
                search_term in email['snippet'].lower()):
                filtered_emails.append(email)
        
        # Display results
        if filtered_emails:
            self.update_status(f"Found {len(filtered_emails)} matching emails")
            self.display_filtered_emails(filtered_emails)
        else:
            self.update_status(f"No emails found containing '{search_term}'")
            # Create a "no results" message
            no_results_label = ctk.CTkLabel(
                self.email_list_inner, 
                text=f"No emails found containing\n'{search_term}'",
                font=ctk.CTkFont(size=14),
                anchor="center",
                height=100
            )
            no_results_label.pack(pady=20, padx=10)
    
    def clear_search(self):
        """Clear search and show all emails"""
        self.search_entry.delete(0, 'end')
        self.clear_search_button.configure(state="disabled")
        
        # Clear email list
        for widget in self.email_list_inner.winfo_children():
            widget.destroy()
        
        # Display all emails
        if self.emails:
            self.populate_email_list()
            self.update_status(f"Showing all {len(self.emails)} emails")
        else:
            self.update_status("No emails loaded")
    
    def display_filtered_emails(self, filtered_emails):
        """Display only the filtered emails with 3-dot menu"""
        for i, email in enumerate(filtered_emails):
            # Create a frame for each email item
            email_frame = ctk.CTkFrame(
                self.email_list_inner, 
                corner_radius=8,
                fg_color=("gray80", "gray20") if email['id'] == self.selected_email_id else ("gray85", "gray25"),
                border_width=1,
                border_color=("gray70", "gray40")
            )
            email_frame.pack(fill="x", padx=8, pady=5)
            
            # Configure grid for email item layout
            email_frame.columnconfigure(0, weight=0)  # For unread indicator
            email_frame.columnconfigure(1, weight=1)  # For content
            email_frame.columnconfigure(2, weight=0)  # For menu button
            
            # Add unread indicator if email hasn't been viewed
            content_column = 1
            if email['id'] not in self.viewed_email_ids:
                unread_indicator = ctk.CTkLabel(
                    email_frame, 
                    text="●", 
                    font=ctk.CTkFont(size=12),
                    text_color=("#4a88cf", "#3a7ebf"),  # Blue color for unread indicator
                    width=10
                )
                unread_indicator.grid(row=0, column=0, rowspan=3, padx=(5,0), pady=5)
            else:
                # Add empty space for alignment
                spacer = ctk.CTkLabel(email_frame, text="", width=10)
                spacer.grid(row=0, column=0, rowspan=3, padx=(5,0), pady=5)
            
            # Gmail preview with sender, subject, and snippet
            sender_text = email['sender'].split('<')[0].strip()
            if len(sender_text) > 25:
                sender_text = sender_text[:22] + "..."
                
            sender_label = ctk.CTkLabel(email_frame, text=sender_text, 
                                    font=ctk.CTkFont(weight="bold"),
                                    anchor="w")
            sender_label.grid(row=0, column=content_column, sticky="w", padx=5, pady=(5, 0))
            
            subject_text = email['subject']
            if len(subject_text) > 40:
                subject_text = subject_text[:37] + "..."
                
            subject_label = ctk.CTkLabel(email_frame, text=subject_text, anchor="w")
            subject_label.grid(row=1, column=content_column, sticky="w", padx=5, pady=(2, 0))
            
            snippet_text = email['snippet']
            if len(snippet_text) > 60:
                snippet_text = snippet_text[:57] + "..."
                
            snippet_label = ctk.CTkLabel(email_frame, text=snippet_text, 
                                    font=ctk.CTkFont(size=12),
                                    text_color="gray50", anchor="w")
            snippet_label.grid(row=2, column=content_column, sticky="w", padx=5, pady=(2, 5))
            
            # Add 3-dot menu button
            menu_button = ctk.CTkButton(
                email_frame,
                text="⋮",
                width=20,
                height=20,
                corner_radius=6,
                fg_color="transparent",
                text_color=("gray40", "gray80"),
                hover_color=("gray80", "gray30"),
                border_width=0,
                command=lambda id=email['id']: self.show_email_menu(id)
            )
            menu_button.grid(row=0, column=2, padx=5, pady=5, sticky="ne")
            
            # Store email ID in the frame for retrieval when clicked
            email_frame.email_id = email['id']
            
            # Bind click event to view email detail
            for widget in [email_frame, sender_label, subject_label, snippet_label]:
                widget.bind("<Button-1>", lambda e, id=email['id']: self.fetch_and_display_email_detail(id))
            
            if email['id'] not in self.viewed_email_ids:
                unread_indicator.bind("<Button-1>", lambda e, id=email['id']: self.fetch_and_display_email_detail(id))
            
            # Change cursor to hand when hovering over email items
            for widget in [email_frame, sender_label, subject_label, snippet_label]:
                widget.bind("<Enter>", lambda e, w=widget: self.on_hover_enter(w))
                widget.bind("<Leave>", lambda e, w=widget: self.on_hover_leave(w))
            
            if email['id'] not in self.viewed_email_ids:
                unread_indicator.bind("<Enter>", lambda e, w=unread_indicator: self.on_hover_enter(w))
                unread_indicator.bind("<Leave>", lambda e, w=unread_indicator: self.on_hover_leave(w))

    def show_email_menu(self, email_id):
        """Show popup menu for email options"""
        # Create a popup menu
        menu = tk.Menu(self, tearoff=0)
        
        # Add menu options
        menu.add_command(label="Mark as Spam", command=lambda: self.mark_as_spam(email_id))
        
        # Find the menu button that was clicked
        for frame in self.email_list_inner.winfo_children():
            if hasattr(frame, 'email_id') and frame.email_id == email_id:
                for widget in frame.winfo_children():
                    if isinstance(widget, ctk.CTkButton) and widget.cget("text") == "⋮":
                        # Get the absolute position of the menu button
                        x = widget.winfo_rootx()
                        y = widget.winfo_rooty() + widget.winfo_height()
                        
                        # Display the menu at the button position
                        menu.tk_popup(x, y)
                        break
                break
        
        # Make sure to grab the focus back when menu is closed
        menu.bind("<FocusOut>", lambda event: menu.unpost())

    
    def display_welcome_message(self):
        """Display welcome message in the email content area"""
        self.subject_label.configure(text="Logistic Regression System")
        self.from_label.configure(text="")
        self.to_label.configure(text="")
        self.date_label.configure(text="")
        
        welcome_text = """
        Welcome to the Logistic Regression System!

        To get started, click the "Fetch Emails" button in the sidebar.

        """
        self.email_content.delete("1.0", "end")
        self.email_content.insert("end", welcome_text)
    
    def resize_canvas_frame(self, event):
        # Resize the frame inside the canvas when canvas size changes
        self.canvas.itemconfig(self.canvas_frame, width=event.width)
    
    def open_google_cloud(self):
        """Open Google Cloud Console in browser"""
        self.status_var.set("Opening Google Cloud Console...")
        webbrowser.open("https://console.cloud.google.com/apis/credentials")
    
    def start_fetch_thread(self):
        """Start fetching emails in a separate thread to avoid UI freezing"""
        self.fetch_button.configure(state="disabled")
        self.status_var.set("Connecting to Gmail...")
        
        # Start thread
        thread = threading.Thread(target=self.fetch_and_display_emails)
        thread.daemon = True
        thread.start()

    def toggle_mode(self):
        if self.mode_toggle.get() == 1:
            ctk.set_appearance_mode("Dark")
            self.email_content.config(bg='#2b2b2b', fg='white')
        else:
            ctk.set_appearance_mode("Light")
            self.email_content.config(bg='white', fg='black')
        
        # Update canvas background color when mode changes
        self.canvas.configure(bg=self.cget("fg_color")[1 if ctk.get_appearance_mode() == "Dark" else 0])

    def fetch_and_display_emails(self):
        """Fetch emails and display them in the email list."""
        self.viewed_email_ids = set()  # Reset viewed emails
        try:
            # Clear previous email list
            for widget in self.email_list_inner.winfo_children():
                widget.destroy()
            
            # Clear email detail view
            self.subject_label.configure(text="")
            self.from_label.configure(text="")
            self.to_label.configure(text="")
            self.date_label.configure(text="")
            self.email_content.delete("1.0", "end")
            
            # Reset prediction results
            self.reset_prediction_display()
            
            # Disable predict button until an email is selected
            self.predict_button.configure(state="disabled")
            
            self.update_status("Authenticating with Gmail...")
            
            self.gmail_service = authenticate_gmail()
            
            if not self.gmail_service:
                self.update_status("Authentication failed. Check console for details.")
                self.display_auth_error()
                self.enable_fetch_button()
                return
            
            self.update_status("Fetching emails...")
            self.emails = fetch_emails(self.gmail_service)

            if not self.emails:
                self.update_status("No emails found.")
                self.email_content.insert("end", "No emails found or error fetching emails.\n")
                self.enable_fetch_button()
                return

            self.update_status(f"Found {len(self.emails)} emails...")
            self.populate_email_list()
            
            self.update_status(f"Successfully loaded {len(self.emails)} emails")
        except Exception as e:
            self.update_status(f"Error: {str(e)}")
            self.email_content.insert("end", f"An error occurred: {str(e)}\n")
        finally:
            self.enable_fetch_button()
    
    def display_auth_error(self):
        """Display authentication error message in the content area"""
        self.email_content.insert("end", "ERROR: Could not authenticate with Gmail.\n\n")
        self.email_content.insert("end", "Possible issues:\n")
        self.email_content.insert("end", "1. Your email might not be added as a test user in Google Cloud Console\n")
        self.email_content.insert("end", "2. The credentials file might be invalid or missing\n")
        self.email_content.insert("end", "3. OAuth consent screen might not be set up correctly\n\n")
        self.email_content.insert("end", "Click 'Setup Gmail API' button for instructions.")
    
    def populate_email_list(self):
        """Populate the email list with all fetched emails"""
        # Clear previous email list
        for widget in self.email_list_inner.winfo_children():
            widget.destroy()
            
        # Display all emails
        self.display_filtered_emails(self.emails)
    
    def on_hover_enter(self, widget):
        """Change cursor to hand on hover"""
        widget.configure(cursor="hand2")
        if isinstance(widget, ctk.CTkFrame):
            widget.configure(fg_color=("gray90", "gray30"))
    
    def on_hover_leave(self, widget):
        """Restore cursor on leave"""
        widget.configure(cursor="")
        if isinstance(widget, ctk.CTkFrame):
            widget.configure(fg_color=("gray75", "gray25") if widget.email_id != self.selected_email_id else ("gray80", "gray20"))
    
    def fetch_and_display_email_detail(self, email_id):
        """Fetch and display the full content of the selected email"""
        if not self.gmail_service or not email_id:
            return
                
        self.selected_email_id = email_id
        
        # Mark this email as viewed
        self.viewed_email_ids.add(email_id)
        
        # Update status
        self.update_status(f"Loading email...")
        
        # Highlight the selected email in the list and remove highlight from others
        for frame in self.email_list_inner.winfo_children():
            if hasattr(frame, 'email_id'):
                if frame.email_id == email_id:
                    frame.configure(fg_color=("gray80", "gray20"))
                    
                    # Replace blue dot with empty spacer if it exists
                    for child in frame.winfo_children():
                        if isinstance(child, ctk.CTkLabel) and child.cget("text") == "●":
                            child.destroy()
                            # Add spacer instead
                            spacer = ctk.CTkLabel(frame, text="", width=10)
                            spacer.grid(row=0, column=0, rowspan=3, padx=(5,0), pady=5)
                else:
                    frame.configure(fg_color=("gray75", "gray25"))
        
        # Reset prediction display when a new email is selected
        self.reset_prediction_display()
        
        # Start thread to fetch full email
        thread = threading.Thread(target=self.load_email_content, args=(email_id,))
        thread.daemon = True
        thread.start()
    
    def reset_prediction_display(self):
        """Reset the prediction display to its default state"""
        self.prediction_label.configure(text="Detection: Not analyzed")
        self.confidence_label.configure(text="Confidence: 0%")
        self.prediction_frame.configure(fg_color=self.cget("fg_color"))  # Reset background color
    
    def load_email_content(self, email_id):
        """Load the full content of an email in a separate thread"""
        try:
            # Get full email details
            email = get_email_by_id(self.gmail_service, email_id)
            
            if not email:
                self.update_status("Failed to load email content.")
                return
                
            # Update UI with email content
            self.after(0, lambda: self.update_email_detail_ui(email))
            
            self.update_status("Email loaded successfully")
        except Exception as e:
            self.update_status(f"Error loading email: {str(e)}")
    
    def update_email_detail_ui(self, email):
        """Update the email detail UI with the provided email data"""
        # Update header information
        self.subject_label.configure(text=email['subject'])
        self.from_label.configure(text=f"From: {email['sender']}")
        self.to_label.configure(text=f"To: {email['to']}")
        self.date_label.configure(text=f"Date: {email['date']}")
        
        # Update content
        self.email_content.delete("1.0", "end")
        self.email_content.insert("end", email['body'])
        
        # Store the current email content for spam prediction
        self.current_email_content = email['body']

        # Configure font for the email content
        self.email_content.tag_configure("content_font", font=("Arial", 12))
        self.email_content.tag_add("content_font", "1.0", "end")
        
        # Enable the predict button now that an email is loaded
        self.predict_button.configure(state="normal")

    def on_folder_hover_enter(self, widget):
        """Change cursor to hand on folder hover"""
        widget.configure(cursor="hand2")
        if isinstance(widget, ctk.CTkFrame):
            widget.configure(fg_color=("gray90", "gray30"))

    def on_folder_hover_leave(self, widget):
        """Restore cursor on folder leave"""
        widget.configure(cursor="")
        if isinstance(widget, ctk.CTkFrame):
            widget.configure(fg_color="transparent")

    def show_spam_folder(self, event=None):
        """Show only spam-marked emails"""
        if not hasattr(self, 'spam_emails'):
            self.update_status("No spam emails to display")
            return
            
        # Clear previous email list
        for widget in self.email_list_inner.winfo_children():
            widget.destroy()
            
        if not self.spam_emails:
            # Show empty state
            no_spam_label = ctk.CTkLabel(
                self.email_list_inner, 
                text="No spam emails",
                font=ctk.CTkFont(size=14),
                anchor="center",
                height=100
            )
            no_spam_label.pack(pady=20, padx=10)
            self.update_status("Spam folder is empty")
        else:
            # Display spam emails
            self.display_filtered_emails(self.spam_emails)
            self.update_status(f"Showing {len(self.spam_emails)} spam emails")

    def show_inbox_folder(self, event=None):
        """Show all non-spam emails"""
        if not self.emails:
            self.update_status("No emails loaded")
            return
            
        # Clear previous email list
        for widget in self.email_list_inner.winfo_children():
            widget.destroy()
            
        # Get non-spam emails
        if hasattr(self, 'spam_emails') and self.spam_emails:
            spam_ids = [email['id'] for email in self.spam_emails]
            inbox_emails = [email for email in self.emails if email['id'] not in spam_ids]
        else:
            inbox_emails = self.emails
            
        # Display inbox emails
        self.display_filtered_emails(inbox_emails)
        self.update_status(f"Showing {len(inbox_emails)} inbox emails")

    def mark_as_spam(self, email_id):
        """Mark an email as spam and move it to the spam folder"""
        if not hasattr(self, 'spam_emails'):
            self.spam_emails = []
            
        # Find the email in the emails list
        email = next((e for e in self.emails if e['id'] == email_id), None)
        if not email:
            self.update_status("Email not found")
            return
            
        # Check if already in spam folder
        if any(e['id'] == email_id for e in self.spam_emails):
            self.update_status("Email already marked as spam")
            return
            
        # Add to spam folder
        self.spam_emails.append(email)
        self.update_status(f"Email marked as spam")
        
        # If we're viewing the inbox, refresh it
        if all(not isinstance(w, ctk.CTkLabel) or w.cget("text") != "No spam emails" 
            for w in self.email_list_inner.winfo_children()):
            self.show_inbox_folder()

    
    def predict_spam(self):
        """Analyze the current email content for spam"""
        if not self.current_email_content:
            self.update_status("No email content to analyze")
            return
        
        self.update_status("Analyzing email for spam...")
        
        # Disable predict button while analyzing
        self.predict_button.configure(state="disabled")
        
        # Run prediction in a separate thread to keep UI responsive
        thread = threading.Thread(target=self.run_prediction)
        thread.daemon = True
        thread.start()
    
    def run_prediction(self):
        """Run the spam prediction in a separate thread"""
        try:
            # Make sure you're using the right function with the right parameters
            label, confidence = predict_spam(self.current_email_content)
            
            # If using predict_spam directly, multiply confidence by 100 for display
            if isinstance(confidence, float) and confidence <= 1.0:
                confidence = confidence * 100
                
            # Update UI with prediction result
            self.after(0, lambda: self.show_prediction_result(label, confidence))
            
            self.update_status("Email analysis complete")
        except Exception as e:
            self.update_status(f"Error analyzing email: {str(e)}")
            # Re-enable the predict button
            self.after(0, lambda: self.predict_button.configure(state="normal"))
    
    def show_prediction_result(self, label, confidence):
        # Format the confidence as percentage
        if isinstance(confidence, float):
            confidence_text = f"{confidence:.1f}%"
        else:
            confidence_text = f"{confidence}"
        
        # Update labels with modern icons
        if label.lower() == "spam":
            icon = "⚠️"  # Modern stop icon
            bg_color = ("#FFE6E6", "#5E3A3A")  # Lighter red in light mode, darker in dark
            border_color = ("#FFB6B6", "#804040")
        else:
            icon = "✅"  # Modern checkmark
            bg_color = ("#E6FFE6", "#3A5E3A")  # Lighter green in light mode, darker in dark
            border_color = ("#B6FFB6", "#408040")
        
        self.prediction_label.configure(text=f"{icon} Detection: {label}", anchor="e", justify="right")
        self.confidence_label.configure(text=f"Confidence: {confidence_text}", anchor="e", justify="right")
        
        # Set background color based on prediction
        self.prediction_frame.configure(
            fg_color=bg_color,
            border_color=border_color,
        )
        
        # Re-enable the predict button
        self.predict_button.configure(state="normal")
    
    def update_status(self, message):
        """Update status bar - can be called from any thread"""
        self.after(0, lambda: self.status_var.set(f"ℹ️ {message}"))
    
    def enable_fetch_button(self):
        """Re-enable fetch button - can be called from any thread"""
        self.after(0, lambda: self.fetch_button.configure(state="normal"))


# Run the app
if __name__ == "__main__":
    app = SpamPredictionApp()
    app.mainloop()