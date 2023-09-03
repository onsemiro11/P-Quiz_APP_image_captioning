import 'package:flutter/material.dart';
import 'package:install_test/Screen/main_screen.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'I\'m a Teacher',
      theme: ThemeData(fontFamily: 'Raleway'),
      home: MainScreen(),
    );
  }
}
