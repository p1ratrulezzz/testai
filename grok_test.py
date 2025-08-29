#!/usr/bin/env python3
"""
Простой тестовый скрипт для модели Grok-1 от xAI
Загружает модель с Hugging Face и тестирует её с промптом
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
import sys
import os

# Подавляем предупреждения для чистого вывода
warnings.filterwarnings("ignore")

class GrokTester:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "hpcai-tech/grok-1"

    def check_system_requirements(self):
        """Проверка системных требований"""
        print("🔍 Проверка системных требований...")

        # Проверка CUDA
        if not torch.cuda.is_available():
            print("⚠️  ПРЕДУПРЕЖДЕНИЕ: CUDA недоступна. Модель будет работать на CPU (очень медленно)")
            print("   Рекомендуется использовать GPU с минимум 16GB VRAM")
            response = input("Продолжить? (y/n): ")
            if response.lower() != 'y':
                sys.exit(0)
        else:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"✅ GPU найден: {torch.cuda.get_device_name(0)}")
            print(f"   Память GPU: {gpu_memory:.1f} GB")

            if gpu_memory < 16:
                print("⚠️  ПРЕДУПРЕЖДЕНИЕ: GPU память может быть недостаточной для Grok-1")
                print("   Модель требует минимум 16GB VRAM для стабильной работы")

        # Проверка доступного места на диске
        statvfs = os.statvfs('.')
        free_space = statvfs.f_frsize * statvfs.f_bavail / (1024**3)
        print(f"💾 Свободное место на диске: {free_space:.1f} GB")

        if free_space < 50:
            print("⚠️  ПРЕДУПРЕЖДЕНИЕ: Недостаточно места на диске")
            print("   Модель требует около 314GB места")

    def load_model(self):
        """Загрузка модели и токенизатора"""
        try:
            print(f"\n📥 Загрузка токенизатора для {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            print(f"📥 Загрузка модели {self.model_name}...")
            print("   ⏳ Это может занять значительное время (модель весит ~314GB)...")

            # Загружаем модель с оптимизациями
            torch.set_default_dtype(torch.bfloat16)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )

            if self.device == "cpu" and torch.cuda.is_available():
                print("   🔄 Перемещение модели на GPU...")
                self.model = self.model.to(self.device)

            self.model.eval()
            print("✅ Модель успешно загружена!")

        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {str(e)}")
            print("\n💡 Возможные решения:")
            print("   1. Убедитесь, что у вас есть доступ к интернету")
            print("   2. Проверьте, что у вас достаточно RAM/GPU памяти")
            print("   3. Попробуйте перезапустить скрипт")
            sys.exit(1)

    def generate_response(self, prompt, max_length=200, temperature=0.7):
        """Генерация ответа на промпт"""
        try:
            print(f"\n🤖 Обработка промпта: '{prompt}'")
            print("   ⏳ Генерация ответа...")

            # Токенизация входного промпта
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
            input_ids = input_ids.to(self.device)
            attention_mask = torch.ones_like(input_ids)

            # Генерация с параметрами
            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_length": max_length,
                "temperature": temperature,
                "do_sample": True,
                "pad_token_id": self.tokenizer.eos_token_id,
                "top_p": 0.9,
                "top_k": 50
            }

            with torch.no_grad():
                outputs = self.model.generate(**inputs)

            # Декодирование результата
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Убираем исходный промпт из ответа
            if response.startswith(prompt):
                response = response[len(prompt):].strip()

            return response

        except Exception as e:
            print(f"❌ Ошибка генерации: {str(e)}")
            return None

    def run_tests(self):
        """Запуск серии тестов"""
        test_prompts = [
            "Что такое искусственный интеллект?",
            "Объясни квантовую физику простыми словами",
            "Напиши короткое стихотворение о программировании",
            "Каковы основные принципы машинного обучения?"
        ]

        print("\n🧪 Запуск тестовых промптов...")
        print("=" * 60)

        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n📝 Тест {i}/{len(test_prompts)}")
            response = self.generate_response(prompt)

            if response:
                print(f"❓ Промпт: {prompt}")
                print(f"🤖 Ответ: {response}")
            else:
                print(f"❌ Не удалось получить ответ для: {prompt}")

            print("-" * 40)

    def interactive_mode(self):
        """Интерактивный режим для пользовательских промптов"""
        print("\n💬 Интерактивный режим (введите 'quit' для выхода)")
        print("=" * 50)

        while True:
            try:
                prompt = input("\n📝 Введите ваш промпт: ")

                if prompt.lower() in ['quit', 'exit', 'выход']:
                    print("👋 До свидания!")
                    break

                if not prompt.strip():
                    continue

                response = self.generate_response(prompt)

                if response:
                    print(f"🤖 Grok-1: {response}")
                else:
                    print("❌ Не удалось получить ответ")

            except KeyboardInterrupt:
                print("\n👋 Программа прервана пользователем")
                break
            except Exception as e:
                print(f"❌ Ошибка: {str(e)}")

def main():
    """Главная функция"""
    print("🚀 Grok-1 Тестер v1.0")
    print("=" * 40)

    # Создаем экземпляр тестера
    tester = GrokTester()

    # Проверяем системные требования
    tester.check_system_requirements()

    # Загружаем модель
    tester.load_model()

    # Выбираем режим работы
    print("\n🎯 Выберите режим работы:")
    print("1. Автоматические тесты")
    print("2. Интерактивный режим")
    print("3. Оба режима")

    try:
        choice = input("\nВведите номер (1-3): ").strip()

        if choice == "1":
            tester.run_tests()
        elif choice == "2":
            tester.interactive_mode()
        elif choice == "3":
            tester.run_tests()
            tester.interactive_mode()
        else:
            print("❌ Некорректный выбор")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n👋 Программа завершена")
    except Exception as e:
        print(f"❌ Критическая ошибка: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
