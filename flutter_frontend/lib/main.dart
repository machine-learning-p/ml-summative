import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'UEMOA Banking Risk Assessment',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        visualDensity: VisualDensity.adaptivePlatformDensity,
        fontFamily: 'Roboto',
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            elevation: 3,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(12),
            ),
          ),
        ),
        cardTheme: CardTheme(
          elevation: 6,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
          ),
        ),
      ),
      home: SplashScreen(),
      debugShowCheckedModeBanner: false,
    );
  }
}

class SplashScreen extends StatefulWidget {
  @override
  _SplashScreenState createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> with TickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _fadeAnimation;
  late Animation<double> _scaleAnimation;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: Duration(seconds: 2),
      vsync: this,
    );

    _fadeAnimation = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(parent: _controller, curve: Curves.easeIn),
    );

    _scaleAnimation = Tween<double>(begin: 0.5, end: 1.0).animate(
      CurvedAnimation(parent: _controller, curve: Curves.elasticOut),
    );

    _controller.forward();

    // Navigate to main screen after 3 seconds
    Future.delayed(Duration(seconds: 3), () {
      Navigator.pushReplacement(
        context,
        PageRouteBuilder(
          pageBuilder: (context, animation, secondaryAnimation) => HomeScreen(),
          transitionsBuilder: (context, animation, secondaryAnimation, child) {
            return FadeTransition(opacity: animation, child: child);
          },
          transitionDuration: Duration(milliseconds: 800),
        ),
      );
    });
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [
              Color(0xFF1565C0),
              Color(0xFF0D47A1),
              Color(0xFF0A3D91),
            ],
          ),
        ),
        child: Center(
          child: FadeTransition(
            opacity: _fadeAnimation,
            child: ScaleTransition(
              scale: _scaleAnimation,
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Container(
                    padding: EdgeInsets.all(20),
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.1),
                      borderRadius: BorderRadius.circular(20),
                      border: Border.all(color: Colors.white.withOpacity(0.2)),
                    ),
                    child: Icon(
                      Icons.account_balance,
                      size: 80,
                      color: Colors.white,
                    ),
                  ),
                  SizedBox(height: 30),
                  Text(
                    'UEMOA Banking',
                    style: TextStyle(
                      fontSize: 32,
                      fontWeight: FontWeight.bold,
                      color: Colors.white,
                      letterSpacing: 1.2,
                    ),
                  ),
                  Text(
                    'Risk Assessment',
                    style: TextStyle(
                      fontSize: 24,
                      fontWeight: FontWeight.w300,
                      color: Colors.white70,
                      letterSpacing: 0.8,
                    ),
                  ),
                  SizedBox(height: 40),
                  Container(
                    padding: EdgeInsets.all(16),
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.1),
                      borderRadius: BorderRadius.circular(50),
                    ),
                    child: CircularProgressIndicator(
                      valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
                      strokeWidth: 3,
                    ),
                  ),
                  SizedBox(height: 20),
                  Text(
                    'Powered by AI',
                    style: TextStyle(
                      fontSize: 16,
                      color: Colors.white60,
                      fontStyle: FontStyle.italic,
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}

class HomeScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [
              Color(0xFFF8F9FA),
              Color(0xFFE3F2FD),
              Color(0xFFBBDEFB),
            ],
          ),
        ),
        child: SafeArea(
          child: SingleChildScrollView(
            child: Padding(
              padding: const EdgeInsets.all(20.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  // Header
                  Container(
                    padding: EdgeInsets.symmetric(vertical: 20),
                    child: Column(
                      children: [
                        Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Icon(
                              Icons.account_balance,
                              size: 32,
                              color: Color(0xFF1565C0),
                            ),
                            SizedBox(width: 12),
                            Text(
                              'UEMOA Banking',
                              style: TextStyle(
                                fontSize: 28,
                                fontWeight: FontWeight.bold,
                                color: Color(0xFF1565C0),
                              ),
                            ),
                          ],
                        ),
                        Text(
                          'Risk Assessment Platform',
                          style: TextStyle(
                            fontSize: 16,
                            color: Colors.grey[600],
                            fontWeight: FontWeight.w300,
                          ),
                        ),
                      ],
                    ),
                  ),

                  SizedBox(height: 20),

                  // Main Feature Card
                  Container(
                    decoration: BoxDecoration(
                      gradient: LinearGradient(
                        begin: Alignment.topLeft,
                        end: Alignment.bottomRight,
                        colors: [Colors.white, Colors.blue[50]!],
                      ),
                      borderRadius: BorderRadius.circular(20),
                      boxShadow: [
                        BoxShadow(
                          color: Colors.black.withOpacity(0.1),
                          blurRadius: 20,
                          offset: Offset(0, 10),
                        ),
                      ],
                    ),
                    child: Padding(
                      padding: const EdgeInsets.all(24.0),
                      child: Column(
                        children: [
                          Container(
                            padding: EdgeInsets.all(16),
                            decoration: BoxDecoration(
                              color: Color(0xFF1565C0).withOpacity(0.1),
                              borderRadius: BorderRadius.circular(16),
                            ),
                            child: Icon(
                              Icons.analytics_outlined,
                              size: 60,
                              color: Color(0xFF1565C0),
                            ),
                          ),
                          SizedBox(height: 20),
                          Text(
                            'AI-Powered Risk Analysis',
                            style: TextStyle(
                              fontSize: 24,
                              fontWeight: FontWeight.bold,
                              color: Color(0xFF1565C0),
                            ),
                          ),
                          SizedBox(height: 12),
                          Text(
                            'Predict bank financial health using advanced machine learning algorithms trained on UEMOA banking data',
                            textAlign: TextAlign.center,
                            style: TextStyle(
                              fontSize: 16,
                              color: Colors.grey[700],
                              height: 1.4,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),

                  SizedBox(height: 30),

                  // Action Buttons
                  _buildActionButton(
                    context,
                    icon: Icons.trending_up,
                    title: 'Start Risk Assessment',
                    subtitle: 'Analyze banking metrics',
                    color: Color(0xFF1565C0),
                    onPressed: () {
                      Navigator.push(
                        context,
                        PageRouteBuilder(
                          pageBuilder: (context, animation, secondaryAnimation) => PredictionScreen(),
                          transitionsBuilder: (context, animation, secondaryAnimation, child) {
                            return SlideTransition(
                              position: Tween<Offset>(
                                begin: Offset(1.0, 0.0),
                                end: Offset.zero,
                              ).animate(animation),
                              child: child,
                            );
                          },
                        ),
                      );
                    },
                  ),

                  SizedBox(height: 16),

                  _buildActionButton(
                    context,
                    icon: Icons.info_outline,
                    title: 'About the Model',
                    subtitle: 'Learn about Z-Score predictions',
                    color: Color(0xFF455A64),
                    onPressed: () {
                      Navigator.push(
                        context,
                        PageRouteBuilder(
                          pageBuilder: (context, animation, secondaryAnimation) => InfoScreen(),
                          transitionsBuilder: (context, animation, secondaryAnimation, child) {
                            return SlideTransition(
                              position: Tween<Offset>(
                                begin: Offset(1.0, 0.0),
                                end: Offset.zero,
                              ).animate(animation),
                              child: child,
                            );
                          },
                        ),
                      );
                    },
                  ),

                  SizedBox(height: 30),

                  // Stats Cards
                  Row(
                    children: [
                      Expanded(
                        child: _buildStatCard('742', 'Banks\nAnalyzed', Icons.account_balance),
                      ),
                      SizedBox(width: 16),
                      Expanded(
                        child: _buildStatCard('8', 'UEMOA\nCountries', Icons.flag),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildActionButton(
      BuildContext context, {
        required IconData icon,
        required String title,
        required String subtitle,
        required Color color,
        required VoidCallback onPressed,
      }) {
    return Container(
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: color.withOpacity(0.3),
            blurRadius: 12,
            offset: Offset(0, 6),
          ),
        ],
      ),
      child: ElevatedButton(
        onPressed: onPressed,
        style: ElevatedButton.styleFrom(
          backgroundColor: color,
          foregroundColor: Colors.white,
          padding: EdgeInsets.all(20),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
          ),
          elevation: 0,
        ),
        child: Row(
          children: [
            Container(
              padding: EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.white.withOpacity(0.2),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Icon(icon, size: 24),
            ),
            SizedBox(width: 16),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    title,
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  SizedBox(height: 4),
                  Text(
                    subtitle,
                    style: TextStyle(
                      fontSize: 14,
                      color: Colors.white.withOpacity(0.8),
                    ),
                  ),
                ],
              ),
            ),
            Icon(Icons.arrow_forward_ios, size: 16),
          ],
        ),
      ),
    );
  }

  Widget _buildStatCard(String number, String label, IconData icon) {
    return Container(
      padding: EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 10,
            offset: Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        children: [
          Icon(icon, size: 32, color: Color(0xFF1565C0)),
          SizedBox(height: 12),
          Text(
            number,
            style: TextStyle(
              fontSize: 24,
              fontWeight: FontWeight.bold,
              color: Color(0xFF1565C0),
            ),
          ),
          SizedBox(height: 4),
          Text(
            label,
            textAlign: TextAlign.center,
            style: TextStyle(
              fontSize: 12,
              color: Colors.grey[600],
              height: 1.2,
            ),
          ),
        ],
      ),
    );
  }
}

class PredictionScreen extends StatefulWidget {
  @override
  _PredictionScreenState createState() => _PredictionScreenState();
}

class _PredictionScreenState extends State<PredictionScreen> with TickerProviderStateMixin {
  final _formKey = GlobalKey<FormState>();
  bool _isLoading = false;
  Map<String, dynamic>? _predictionResult;
  String? _errorMessage;
  int _currentStep = 0;

  late AnimationController _animationController;
  late Animation<double> _fadeAnimation;

  // Controllers for all input fields
  final TextEditingController _countriesNumController = TextEditingController();
  final TextEditingController _yearController = TextEditingController();
  final TextEditingController _rirController = TextEditingController();
  final TextEditingController _sfsController = TextEditingController();
  final TextEditingController _infController = TextEditingController();
  final TextEditingController _eraController = TextEditingController();
  final TextEditingController _inlController = TextEditingController();
  final TextEditingController _debtController = TextEditingController();
  final TextEditingController _sizeController = TextEditingController();
  final TextEditingController _ccController = TextEditingController();
  final TextEditingController _geController = TextEditingController();
  final TextEditingController _psController = TextEditingController();
  final TextEditingController _rqController = TextEditingController();
  final TextEditingController _rlController = TextEditingController();
  final TextEditingController _vaController = TextEditingController();
  final TextEditingController _countriesController = TextEditingController();
  final TextEditingController _banksController = TextEditingController();

  final String apiUrl = 'https://ml-summative-xupo.onrender.com/predict';

  @override
  void initState() {
    super.initState();
    _animationController = AnimationController(
      duration: Duration(milliseconds: 500),
      vsync: this,
    );
    _fadeAnimation = Tween<double>(begin: 0.0, end: 1.0).animate(_animationController);
    _animationController.forward();

    // Set default values
    _countriesController.text = 'Benin';
    _banksController.text = 'Default Bank';
    _yearController.text = '2023';
  }

  @override
  void dispose() {
    _animationController.dispose();
    _countriesNumController.dispose();
    _yearController.dispose();
    _rirController.dispose();
    _sfsController.dispose();
    _infController.dispose();
    _eraController.dispose();
    _inlController.dispose();
    _debtController.dispose();
    _sizeController.dispose();
    _ccController.dispose();
    _geController.dispose();
    _psController.dispose();
    _rqController.dispose();
    _rlController.dispose();
    _vaController.dispose();
    _countriesController.dispose();
    _banksController.dispose();
    super.dispose();
  }

  Future<void> _makePrediction() async {
    if (!_formKey.currentState!.validate()) {
      return;
    }

    setState(() {
      _isLoading = true;
      _predictionResult = null;
      _errorMessage = null;
    });

    try {
      final response = await http.post(
        Uri.parse(apiUrl),
        headers: {
          'Content-Type': 'application/json',
        },
        body: json.encode({
          'countries_num': int.parse(_countriesNumController.text),
          'year': int.parse(_yearController.text),
          'rir': double.parse(_rirController.text),
          'sfs': double.parse(_sfsController.text),
          'inf': double.parse(_infController.text),
          'era': double.parse(_eraController.text),
          'inl': double.parse(_inlController.text),
          'debt': double.parse(_debtController.text),
          'size': double.parse(_sizeController.text),
          'cc': double.parse(_ccController.text),
          'ge': double.parse(_geController.text),
          'ps': double.parse(_psController.text),
          'rq': double.parse(_rqController.text),
          'rl': double.parse(_rlController.text),
          'va': double.parse(_vaController.text),
          'countries': _countriesController.text,
          'banks': _banksController.text,
        }),
      );

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        setState(() {
          _predictionResult = data;
        });
      } else {
        setState(() {
          _errorMessage = 'Error: ${response.statusCode} - ${response.body}';
        });
      }
    } catch (e) {
      setState(() {
        _errorMessage = 'Network error: $e';
      });
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  Widget _buildInputField(
      String label,
      TextEditingController controller,
      String hint, {
        double? min,
        double? max,
        bool isInteger = false,
        IconData? icon,
      }) {
    return Container(
      margin: EdgeInsets.symmetric(vertical: 8),
      child: TextFormField(
        controller: controller,
        decoration: InputDecoration(
          labelText: label,
          hintText: hint,
          prefixIcon: icon != null ? Icon(icon, color: Color(0xFF1565C0)) : null,
          border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(12),
            borderSide: BorderSide(color: Colors.grey[300]!),
          ),
          enabledBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(12),
            borderSide: BorderSide(color: Colors.grey[300]!),
          ),
          focusedBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(12),
            borderSide: BorderSide(color: Color(0xFF1565C0), width: 2),
          ),
          filled: true,
          fillColor: Colors.grey[50],
          labelStyle: TextStyle(color: Colors.grey[700]),
        ),
        keyboardType: isInteger
            ? TextInputType.number
            : TextInputType.numberWithOptions(decimal: true),
        validator: (value) {
          if (value == null || value.isEmpty) {
            return 'Please enter $label';
          }

          if (isInteger) {
            final intValue = int.tryParse(value);
            if (intValue == null) {
              return 'Please enter a valid integer';
            }
            if (min != null && intValue < min) {
              return 'Value must be at least ${min.toInt()}';
            }
            if (max != null && intValue > max) {
              return 'Value must be at most ${max.toInt()}';
            }
          } else {
            final doubleValue = double.tryParse(value);
            if (doubleValue == null) {
              return 'Please enter a valid number';
            }
            if (min != null && doubleValue < min) {
              return 'Value must be at least $min';
            }
            if (max != null && doubleValue > max) {
              return 'Value must be at most $max';
            }
          }
          return null;
        },
      ),
    );
  }

  Widget _buildSectionCard(String title, IconData icon, List<Widget> children) {
    return Container(
      margin: EdgeInsets.symmetric(vertical: 8),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 10,
            offset: Offset(0, 4),
          ),
        ],
      ),
      child: Padding(
        padding: EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(icon, color: Color(0xFF1565C0), size: 24),
                SizedBox(width: 12),
                Text(
                  title,
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                    color: Color(0xFF1565C0),
                  ),
                ),
              ],
            ),
            SizedBox(height: 16),
            ...children,
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Color(0xFFF8F9FA),
      appBar: AppBar(
        title: Text(
          'Banking Risk Prediction',
          style: TextStyle(fontWeight: FontWeight.bold),
        ),
        backgroundColor: Color(0xFF1565C0),
        foregroundColor: Colors.white,
        elevation: 0,
        flexibleSpace: Container(
          decoration: BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
              colors: [Color(0xFF1565C0), Color(0xFF0D47A1)],
            ),
          ),
        ),
      ),
      body: FadeTransition(
        opacity: _fadeAnimation,
        child: Form(
          key: _formKey,
          child: ListView(
            padding: EdgeInsets.all(16),
            children: [
              // Basic Information
              _buildSectionCard(
                'Basic Information',
                Icons.info_outline,
                [
                  _buildInputField(
                    'Country Number',
                    _countriesNumController,
                    'Enter 1-8 for UEMOA countries',
                    min: 1,
                    max: 8,
                    isInteger: true,
                    icon: Icons.numbers,
                  ),
                  _buildInputField(
                    'Year',
                    _yearController,
                    'Enter year (2010-2025)',
                    min: 2010,
                    max: 2025,
                    isInteger: true,
                    icon: Icons.calendar_today,
                  ),
                  _buildInputField(
                    'Country Name',
                    _countriesController,
                    'Enter country name',
                    icon: Icons.flag,
                  ),
                  _buildInputField(
                    'Bank Name',
                    _banksController,
                    'Enter bank name',
                    icon: Icons.account_balance,
                  ),
                ],
              ),

              // Risk Metrics
              _buildSectionCard(
                'Risk Metrics',
                Icons.warning_amber_outlined,
                [
                  _buildInputField(
                    'Risk Index Rating (RIR)',
                    _rirController,
                    'Enter 0-10',
                    min: 0,
                    max: 10,
                    icon: Icons.trending_down,
                  ),
                  _buildInputField(
                    'Economic Risk Assessment (ERA)',
                    _eraController,
                    'Enter 0-10',
                    min: 0,
                    max: 10,
                    icon: Icons.assessment,
                  ),
                  _buildInputField(
                    'Inflation Rate (INF)',
                    _infController,
                    'Enter -5 to 20',
                    min: -5,
                    max: 20,
                    icon: Icons.trending_up,
                  ),
                ],
              ),

              // Financial Metrics
              _buildSectionCard(
                'Financial Metrics',
                Icons.monetization_on_outlined,
                [
                  _buildInputField(
                    'Solvency & Financial Stability (SFS)',
                    _sfsController,
                    'Enter 0-100',
                    min: 0,
                    max: 100,
                    icon: Icons.security,
                  ),
                  _buildInputField(
                    'Debt Level (DEBT)',
                    _debtController,
                    'Enter 0-100',
                    min: 0,
                    max: 100,
                    icon: Icons.credit_card,
                  ),
                  _buildInputField(
                    'Bank Size (SIZE)',
                    _sizeController,
                    'Enter 0-30',
                    min: 0,
                    max: 30,
                    icon: Icons.business,
                  ),
                  _buildInputField(
                    'Capital Adequacy (CC)',
                    _ccController,
                    'Enter 0-100',
                    min: 0,
                    max: 100,
                    icon: Icons.account_balance_wallet,
                  ),
                ],
              ),

              // Performance Metrics
              _buildSectionCard(
                'Performance & Governance',
                Icons.trending_up,
                [
                  _buildInputField(
                    'Governance & Ethics (GE)',
                    _geController,
                    'Enter 0-100',
                    min: 0,
                    max: 100,
                    icon: Icons.gavel,
                  ),
                  _buildInputField(
                    'Profitability & Sustainability (PS)',
                    _psController,
                    'Enter 0-100',
                    min: 0,
                    max: 100,
                    icon: Icons.show_chart,
                  ),
                  _buildInputField(
                    'Regulatory Compliance (RQ)',
                    _rqController,
                    'Enter 0-100',
                    min: 0,
                    max: 100,
                    icon: Icons.rule,
                  ),
                  _buildInputField(
                    'Liquidity Risk (RL)',
                    _rlController,
                    'Enter 0-100',
                    min: 0,
                    max: 100,
                    icon: Icons.water_drop,
                  ),
                  _buildInputField(
                    'Value Added (VA)',
                    _vaController,
                    'Enter 0-100',
                    min: 0,
                    max: 100,
                    icon: Icons.add_circle,
                  ),
                  _buildInputField(
                    'Internationalization Level (INL)',
                    _inlController,
                    'Enter 0-50',
                    min: 0,
                    max: 50,
                    icon: Icons.public,
                  ),
                ],
              ),

              SizedBox(height: 24),

              // Predict Button
              Container(
                height: 60,
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(16),
                  gradient: LinearGradient(
                    colors: [Color(0xFF1565C0), Color(0xFF0D47A1)],
                  ),
                  boxShadow: [
                    BoxShadow(
                      color: Color(0xFF1565C0).withOpacity(0.4),
                      blurRadius: 12,
                      offset: Offset(0, 6),
                    ),
                  ],
                ),
                child: ElevatedButton(
                  onPressed: _isLoading ? null : _makePrediction,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.transparent,
                    shadowColor: Colors.transparent,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(16),
                    ),
                  ),
                  child: _isLoading
                      ? Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      SizedBox(
                        width: 24,
                        height: 24,
                        child: CircularProgressIndicator(
                          strokeWidth: 3,
                          valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
                        ),
                      ),
                      SizedBox(width: 16),
                      Text(
                        'Analyzing...',
                        style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                      ),
                    ],
                  )
                      : Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Icon(Icons.analytics, size: 24),
                      SizedBox(width: 12),
                      Text(
                        'Predict Risk Level',
                        style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: Colors.white),
                      ),
                    ],
                  ),
                ),
              ),

              SizedBox(height: 24),

              // Results Display
              if (_predictionResult != null)
                Container(
                  decoration: BoxDecoration(
                    gradient: LinearGradient(
                      begin: Alignment.topLeft,
                      end: Alignment.bottomRight,
                      colors: [Color(0xFF4CAF50), Color(0xFF2E7D32)],
                    ),
                    borderRadius: BorderRadius.circular(20),
                    boxShadow: [
                      BoxShadow(
                        color: Color(0xFF4CAF50).withOpacity(0.3),
                        blurRadius: 15,
                        offset: Offset(0, 8),
                      ),
                    ],
                  ),
                  child: Padding(
                    padding: EdgeInsets.all(24),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            Container(
                              padding: EdgeInsets.all(8),
                              decoration: BoxDecoration(
                                color: Colors.white.withOpacity(0.2),
                                borderRadius: BorderRadius.circular(12),
                              ),
                              child: Icon(Icons.check_circle, color: Colors.white, size: 28),
                            ),
                            SizedBox(width: 16),
                            const Expanded(
                              child: Text(
                                'Risk Assessment Complete',
                                style: TextStyle(
                                  fontSize: 20,
                                  fontWeight: FontWeight.bold,
                                  color: Colors.white,
                                ),
                              ),
                            ),
                          ],
                        ),
                        SizedBox(height: 20),
                        _buildResultItem(
                          'Z-Score',
                          _predictionResult!['predicted_zscore']?.toString() ?? 'N/A',
                          Icons.score,
                        ),
                        _buildResultItem(
                          'Risk Level',
                          _predictionResult!['risk_level']?.toString() ?? 'N/A',
                          Icons.speed,
                        ),
                        _buildResultItem(
                          'Assessment',
                          _predictionResult!['interpretation']?.toString() ?? 'N/A',
                          Icons.assessment,
                        ),
                        _buildResultItem(
                          'Confidence',
                          _predictionResult!['model_confidence']?.toString() ?? 'N/A',
                          Icons.verified,
                        ),
                      ],
                    ),
                  ),
                ),

              if (_errorMessage != null)
                Container(
                  decoration: BoxDecoration(
                    gradient: LinearGradient(
                      begin: Alignment.topLeft,
                      end: Alignment.bottomRight,
                      colors: [Color(0xFFF44336), Color(0xFFD32F2F)],
                    ),
                    borderRadius: BorderRadius.circular(16),
                    boxShadow: [
                      BoxShadow(
                        color: Color(0xFFF44336).withOpacity(0.3),
                        blurRadius: 12,
                        offset: Offset(0, 6),
                      ),
                    ],
                  ),
                  child: Padding(
                    padding: EdgeInsets.all(20),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Row(
                          children: [
                            Icon(Icons.error, color: Colors.white, size: 24),
                            SizedBox(width: 12),
                            Text(
                              'Error Occurred',
                              style: TextStyle(
                                fontSize: 18,
                                fontWeight: FontWeight.bold,
                                color: Colors.white,
                              ),
                            ),
                          ],
                        ),
                        SizedBox(height: 12),
                        Text(
                          _errorMessage!,
                          style: TextStyle(
                            fontSize: 16,
                            color: Colors.white.withOpacity(0.9),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildResultItem(String label, String value, IconData icon) {
    return Padding(
      padding: EdgeInsets.symmetric(vertical: 8),
      child: Row(
        children: [
          Icon(icon, color: Colors.white.withOpacity(0.8), size: 20),
          SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  label,
                  style: TextStyle(
                    fontSize: 14,
                    color: Colors.white.withOpacity(0.8),
                    fontWeight: FontWeight.w500,
                  ),
                ),
                SizedBox(height: 2),
                Text(
                  value,
                  style: TextStyle(
                    fontSize: 16,
                    color: Colors.white,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class InfoScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Color(0xFFF8F9FA),
      appBar: AppBar(
        title: Text(
          'About the Model',
          style: TextStyle(fontWeight: FontWeight.bold),
        ),
        backgroundColor: Color(0xFF1565C0),
        foregroundColor: Colors.white,
        elevation: 0,
        flexibleSpace: Container(
          decoration: BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
              colors: [Color(0xFF1565C0), Color(0xFF0D47A1)],
            ),
          ),
        ),
      ),
      body: ListView(
        padding: EdgeInsets.all(16),
        children: [
          // Main Info Card
          Container(
            decoration: BoxDecoration(
              gradient: LinearGradient(
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
                colors: [Colors.white, Color(0xFFF5F5F5)],
              ),
              borderRadius: BorderRadius.circular(20),
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withOpacity(0.1),
                  blurRadius: 20,
                  offset: Offset(0, 10),
                ),
              ],
            ),
            child: Padding(
              padding: EdgeInsets.all(24),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Container(
                        padding: EdgeInsets.all(12),
                        decoration: BoxDecoration(
                          color: Color(0xFF1565C0).withOpacity(0.1),
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: Icon(
                          Icons.psychology,
                          size: 32,
                          color: Color(0xFF1565C0),
                        ),
                      ),
                      SizedBox(width: 16),
                      Expanded(
                        child: Text(
                          'UEMOA Banking Risk Assessment',
                          style: TextStyle(
                            fontSize: 24,
                            fontWeight: FontWeight.bold,
                            color: Color(0xFF1565C0),
                          ),
                        ),
                      ),
                    ],
                  ),
                  SizedBox(height: 20),
                  Text(
                    'This AI-powered application predicts bank financial health using the Z-Score metric for banks in the West African Economic and Monetary Union (UEMOA). Our machine learning model analyzes 20+ banking metrics to provide accurate risk assessments.',
                    style: TextStyle(
                      fontSize: 16,
                      color: Colors.grey[700],
                      height: 1.5,
                    ),
                  ),
                ],
              ),
            ),
          ),

          SizedBox(height: 20),

          // Z-Score Interpretation
          _buildInfoCard(
            'Z-Score Interpretation',
            Icons.analytics,
            [
              _buildScoreItem('≥ 2.0', 'Low Risk', 'Excellent financial health', Color(0xFF4CAF50)),
              _buildScoreItem('1.5-2.0', 'Moderate Risk', 'Good financial health', Color(0xFFFFC107)),
              _buildScoreItem('1.0-1.5', 'High Risk', 'Concerning financial health', Color(0xFFFF9800)),
              _buildScoreItem('< 1.0', 'Very High Risk', 'Poor financial health', Color(0xFFF44336)),
            ],
          ),

          SizedBox(height: 20),

          // UEMOA Countries
          _buildInfoCard(
            'UEMOA Member Countries',
            Icons.flag,
            _buildCountryList(),
          ),

          SizedBox(height: 20),

          // Model Features
          _buildInfoCard(
            'Model Features',
            Icons.tune,
            [
              _buildFeatureItem('742 Banks', 'Comprehensive dataset coverage'),
              _buildFeatureItem('20+ Metrics', 'Financial, risk, and governance indicators'),
              _buildFeatureItem('8 Countries', 'Complete UEMOA region analysis'),
              _buildFeatureItem('AI-Powered', 'Advanced machine learning algorithms'),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildInfoCard(String title, IconData icon, List<Widget> children) {
    return Container(
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 10,
            offset: Offset(0, 4),
          ),
        ],
      ),
      child: Padding(
        padding: EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(icon, color: Color(0xFF1565C0), size: 24),
                SizedBox(width: 12),
                Text(
                  title,
                  style: TextStyle(
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                    color: Color(0xFF1565C0),
                  ),
                ),
              ],
            ),
            SizedBox(height: 16),
            ...children,
          ],
        ),
      ),
    );
  }

  Widget _buildScoreItem(String range, String level, String description, Color color) {
    return Padding(
      padding: EdgeInsets.symmetric(vertical: 8),
      child: Row(
        children: [
          Container(
            width: 60,
            padding: EdgeInsets.symmetric(vertical: 6, horizontal: 8),
            decoration: BoxDecoration(
              color: color.withOpacity(0.1),
              borderRadius: BorderRadius.circular(8),
              border: Border.all(color: color.withOpacity(0.3)),
            ),
            child: Text(
              range,
              textAlign: TextAlign.center,
              style: TextStyle(
                fontSize: 12,
                fontWeight: FontWeight.bold,
                color: color,
              ),
            ),
          ),
          SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  level,
                  style: TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                    color: Colors.grey[800],
                  ),
                ),
                Text(
                  description,
                  style: TextStyle(
                    fontSize: 14,
                    color: Colors.grey[600],
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  List<Widget> _buildCountryList() {
    final countries = [
      'Benin', 'Burkina Faso', 'Côte d\'Ivoire', 'Guinea-Bissau',
      'Mali', 'Niger', 'Senegal', 'Togo'
    ];

    return countries.asMap().entries.map((entry) {
      int index = entry.key;
      String country = entry.value;

      return Padding(
        padding: EdgeInsets.symmetric(vertical: 6),
        child: Row(
          children: [
            Container(
              width: 32,
              height: 32,
              decoration: BoxDecoration(
                color: Color(0xFF1565C0).withOpacity(0.1),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Center(
                child: Text(
                  '${index + 1}',
                  style: TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.bold,
                    color: Color(0xFF1565C0),
                  ),
                ),
              ),
            ),
            SizedBox(width: 16),
            Text(
              country,
              style: TextStyle(
                fontSize: 16,
                color: Colors.grey[700],
              ),
            ),
          ],
        ),
      );
    }).toList();
  }

  Widget _buildFeatureItem(String title, String description) {
    return Padding(
      padding: EdgeInsets.symmetric(vertical: 8),
      child: Row(
        children: [
          Container(
            width: 8,
            height: 8,
            decoration: BoxDecoration(
              color: Color(0xFF1565C0),
              borderRadius: BorderRadius.circular(4),
            ),
          ),
          SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                    color: Colors.grey[800],
                  ),
                ),
                Text(
                  description,
                  style: TextStyle(
                    fontSize: 14,
                    color: Colors.grey[600],
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}