import 'package:flutter/material.dart';
import 'home_page.dart';

void main() {
  runApp(PawScanApp());
}

class PawScanApp extends StatelessWidget {
  const PawScanApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'PawScan',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(primarySwatch: Colors.teal),
      home: HomePage(),
    );
  }
}
