import 'package:flutter/material.dart';

class SelectPhoto extends StatelessWidget {
  final String textLabel;
  final IconData icon;
  final void Function()? onTap;

  const SelectPhoto({
    Key? key,
    required this.textLabel,
    required this.icon,
    required this.onTap,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return ElevatedButton(
      onPressed: onTap,
      style: ElevatedButton.styleFrom(
        elevation: 10, // 버튼의 고도
        primary: Color(0xFF2F3542), // 버튼의 배경색
        shape: const StadiumBorder(), // 버튼의 모양 => 원형
      ),
      child: Padding(
        // 내부 위젯에 여백 설정
        padding: const EdgeInsets.symmetric(
          vertical: 10,
          horizontal: 6,
        ),

        // 아이콘과 텍스트 레이블을 가로로 나열
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              icon,
              color: Colors.white,
            ),
            const SizedBox(
              width: 14,
            ),
            Text(
              textLabel,
              style: const TextStyle(
                fontSize: 16,
                color: Colors.white,
              ),
            )
          ],
        ),
      ),
    );
  }
}
