from flask import render_template, url_for, flash, redirect, request, session
from flask_login import login_user, current_user, logout_user, login_required
from app import app, db
from models import User, ScanResult
from forms import RegistrationForm, LoginForm, UploadForm
from utils import save_uploaded_file, predict_knee_oa, generate_comparison_chart
import os


@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html', title='Knee Osteoarthritis Prediction')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)
        
        db.session.add(user)
        db.session.commit()
        
        flash('Your account has been created! You can now log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html', title='Register', form=form)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        
        if user and user.check_password(form.password.data):
            login_user(user, remember=form.remember.data)
            next_page = request.args.get('next')
            flash('Login successful!', 'success')
            return redirect(next_page if next_page else url_for('dashboard'))
        else:
            flash('Login unsuccessful. Please check your email and password.', 'danger')
    
    return render_template('login.html', title='Login', form=form)


@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('home'))


@app.route('/dashboard')
@login_required
def dashboard():
    # Get user's scan history
    scan_results = ScanResult.query.filter_by(user_id=current_user.id).order_by(ScanResult.scan_date.desc()).all()
    return render_template('dashboard.html', title='Dashboard', scan_results=scan_results)


@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    form = UploadForm()
    
    if form.validate_on_submit():
        # Save the uploaded file
        file_path = save_uploaded_file(form.image.data)
        
        if file_path:
            # Store the file path in session to use in the analysis route
            session['uploaded_image_path'] = file_path
            return redirect(url_for('analyze'))
        else:
            flash('There was an error uploading your file. Please try again.', 'danger')
    
    return render_template('upload.html', title='Upload Scan', form=form)


@app.route('/analyze')
@login_required
def analyze():
    # Get the uploaded file path from session
    file_path = session.get('uploaded_image_path')
    
    if not file_path or not os.path.exists(file_path):
        flash('No image found. Please upload a scan image first.', 'warning')
        return redirect(url_for('upload'))
    
    # Analyze the image
    prediction = predict_knee_oa(file_path)
    
    if not prediction:
        flash('Error analyzing the image. Please try uploading a different image.', 'danger')
        return redirect(url_for('upload'))
    
    # Check if the scan is valid
    if prediction['disease_name'] == 'Not a Valid Scan':
        # Delete the uploaded file if it's invalid
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Deleted invalid image: {file_path}")
        except Exception as e:
            print(f"Error deleting invalid file: {str(e)}")
            
        # Clear the session data to prevent access to the results page
        if 'uploaded_image_path' in session:
            session.pop('uploaded_image_path')
        if 'result_id' in session:
            session.pop('result_id')
            
        flash('INVALID IMAGE: The system only accepts knee radiographs (X-rays) or MRI scans. Please upload a proper medical knee scan. Non-medical images cannot be processed for diagnosis.', 'danger')
        return redirect(url_for('upload'))
    
    # Generate comparison chart
    chart_img = generate_comparison_chart(prediction['knee_health_score'])
    
    # Save the result to database
    result = ScanResult(
        user_id=current_user.id,
        image_path=os.path.basename(file_path),  # Store only filename, not full path
        disease_name=prediction['disease_name'],
        severity_level=prediction['severity_level'],
        confidence=prediction['confidence'],
        knee_health_score=prediction['knee_health_score']
    )
    
    db.session.add(result)
    db.session.commit()
    
    # Store result ID in session for results page
    session['result_id'] = result.id
    
    return redirect(url_for('results'))


@app.route('/results')
@login_required
def results():
    # Get the result ID from session
    result_id = session.get('result_id')
    
    if not result_id:
        flash('No analysis results found. Please upload a scan image first.', 'warning')
        return redirect(url_for('upload'))
    
    # Get the result from database
    result = ScanResult.query.get_or_404(result_id)
    
    # Ensure the result belongs to the current user
    if result.user_id != current_user.id:
        flash('You do not have permission to view this result.', 'danger')
        return redirect(url_for('dashboard'))
        
    # Double-check that we're not accidentally displaying an invalid scan result
    if result.disease_name == 'Not a Valid Scan':
        flash('INVALID IMAGE: The system only accepts knee radiographs (X-rays) or MRI scans. Please upload a proper medical knee scan.', 'danger')
        
        # Remove the result from the session
        if 'result_id' in session:
            session.pop('result_id')
            
        return redirect(url_for('upload'))
    
    # Generate comparison chart
    chart_img = generate_comparison_chart(result.knee_health_score)
    
    # Get relative image path for display
    image_path = result.image_path
    
    return render_template(
        'results.html',
        title='Clinical Analysis',
        result=result,
        chart_img=chart_img,
        image_path=image_path
    )
